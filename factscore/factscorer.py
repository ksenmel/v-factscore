import asyncio
import itertools
from typing import List, Tuple
import numpy as np

from factscore.api_requests import APICompletions
from factscore.atomic_facts import AtomicFactGenerator
from factscore.api_requests import APIEmbeddingFunction
from factscore.database import DocDB

from factscore.database import postprocess_text


class FactScorer:
    def __init__(
        self,
        completions_base_url="https://openrouter.ai/api/v1/chat/completions",
        completions_model_name="mistral/ministral-8b",
        embedding_base_url="http://localhost:11434/api/embeddings",
        embedding_model_name="nomic-embed-text",
    ):
        self.completions_lm = APICompletions(
            base_url=completions_base_url, model_name=completions_model_name
        )
        self.embeddings_lm = APIEmbeddingFunction(
            base_url=embedding_base_url, model_name=embedding_model_name
        )
        self.af_generator = AtomicFactGenerator(llm=self.completions_lm)
        self.db = None

    def register_knowledge_source(
        self,
        faiss_index,
        data_db,
        table_name,
    ):
        self.db = DocDB(
            db_path=data_db,
            faiss_index=faiss_index,
            ef=self.embeddings_lm,
            table=table_name,
        )

    async def get_score(self, topics: list, generations: list, k=5):
        """
        Computes factscore for each generation

        topics: topics of the generations.
        generations: the generations we try to score.
        k: how many articles we want to add to the RAG context.
        """

        scores, decisions = [], []

        for topic, gen in zip(topics, generations):
            facts, _ = await self.af_generator.run(
                gen
            )  # ({sentence: [facts]}, [spans of the paragraphs])
            gen_atoms = list(itertools.chain.from_iterable(facts.values()))

            if len(gen_atoms) == 0:
                scores.append(0)
                decisions.append(None)
                continue

            generation_decisions = await self.is_supported(gen_atoms, [topic], k=k)

            if len(generation_decisions) > 0:
                score = np.mean([d["is_supported"] for d in generation_decisions])
                decisions.append(generation_decisions)
                scores.append(score)

        out = {
            "decisions": decisions,
            "scores": scores,
        }

        return out

    async def is_supported(self, atomic_facts, topic: str = None, k=2):
        """
        Maps `is_supported` boolean label to atoms
        """
        decisions = []
        prompts, retrieves = await self.get_rag_prompts_and_passages(
            atomic_facts, topic, k
        )
        atoms = [p[0] for p in prompts]

        answers = await self.completions_lm.generate([p[1] for p in prompts])

        for answer, atom in zip(answers, atoms):
            generated_answer = answer.lower()

            if "true" in generated_answer:
                is_supported = True
            if "false" in generated_answer:
                is_supported = False
            else:
                continue

            decisions.append({"atom": atom, "is_supported": is_supported})

        print(retrieves)
        return decisions

    async def get_rag_prompts_and_passages(
        self, atomic_facts, topic: str, k: int
    ) -> List[Tuple[str, str]]:
        """
        Returns the retrieval part with appropriate info from wiki for each atomic fact
        """
        prompts = []
        titles, texts = await self.db.search_text_by_queries(queries=atomic_facts, k=k)

        retrieves = []

        postprocess_text(texts)

        for i, atom in enumerate(atomic_facts):
            passages = self.db.get_bm25_passages(topic, atom, texts[i], k=k)

            rag_prompt = f"Task: Answer the question about {topic} based on the given context.\n\n"
            for psg in reversed(passages):
                context = f"Text: {psg}\n"
                rag_prompt += context

            rag_prompt += (
                f'\n\nInput: "{atom.strip()}" True or False? '
                "Answer True if the information is supported by the context above and False otherwise.\n"
                "Output: "
            )

            prompts.append((atom, rag_prompt))

            retrieves.append({"title": titles[i], "passages": passages})

        return prompts, retrieves


if __name__ == "__main__":
    fs = FactScorer()
    fs.register_knowledge_source(faiss_index="", data_db="", table_name="")

    gen = "Albert Einstein was born on March 14, 1879, in the German city of Ulm beside the Danube River. His parents, Hermann Einstein and Pauline Koch, were middle-class secular Jews."

    res = asyncio.run(fs.get_score(topics=["Albert Einstein"], generations=[gen], k=2))

    print(res)
