import asyncio
import os
import sys

from factscore.api_requests import APICompletions
from factscore.atomic_facts import AtomicFactGenerator
from factscore.api_requests import APIEmbeddingFunction
from factscore.database import DocDB

from factscore.database import postprocess_text


class FactScorer:
    def __init__(
        self,
        completions_base_url="https://api.deepinfra.com/v1/openai/chat/completions",
        completions_model_name="Qwen/Qwen2.5-72B-Instruct",
        embedding_base_url="https://api.deepinfra.com/v1/openai/embeddings",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
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
        assert isinstance(topics, list) and isinstance(
            generations, list
        ), "generations and topics must be lists"

        atomic_facts, para_spans = [], dict()

        for topic, gen in zip(topics, generations):
            facts_and_spans = await self.af_generator.run(
                gen
            )  # ({sentence: [facts]}, [spans of the paragraphs])

            generation_atomic_facts, generation_spans = [], dict()
            for triplet in facts_and_spans:
                generation_atomic_facts.extend(triplet[1])
            atomic_facts.append(generation_atomic_facts)

        pass

    async def get_rag_prompts_and_passages(self, atomic_facts, topic: str, k: int):
        """
        Returns the retrieval part with appropriate info from wiki for each atomic fact
        """
        prompts = []
        titles, texts = await self.db.search_text_by_queries(queries=atomic_facts, k=k)

        postprocess_text(texts)

        passages_for_atoms = {}
        for i, atom in enumerate(atomic_facts):
            passages = self.db.get_bm25_passages(topic, atom, texts[i], k=k)
            passages_for_atoms[atom] = passages

            rag_prompt = (
                f"Task: Answer the question about {topic} based on the given context.\n\n"
                f"Title: {titles[i]}\n"
            )
            for psg in reversed(passages):
                context = (
                    f"Text: {psg}\n"
                )
                rag_prompt += context

            rag_prompt += (
                f"\n\nInput: \"{atom.strip()}\" True or False? "
                "Answer True if the information is supported by the context above and False otherwise.\n"
                "Output:"
            )

            print(rag_prompt)

            prompts.append((atom, rag_prompt))
        return prompts, passages_for_atoms


if __name__ == "__main__":
    queries = ["Albert Einstein was a German-born theoretical physicist"]
    fs = FactScorer()
    fs.register_knowledge_source(faiss_index="ivf.index", data_db="/Users/kseniia/enwiki-20230401.db", table_name='documents')

    prompts, passages_for_atoms = asyncio.run(fs.get_rag_prompts_and_passages(queries, "Albert Einstein", 5))
