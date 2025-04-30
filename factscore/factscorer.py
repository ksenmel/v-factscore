import asyncio
import itertools
import numpy as np

from typing import List, Tuple

from factscore.api_requests import APICompletions
from factscore.atomic_facts import AtomicFactGenerator
from factscore.api_requests import APIEmbeddingFunction
from factscore.database import DocDB
from factscore.entities_retriever import EntitiesRetriever


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
        self.entities_retriever = EntitiesRetriever(llm=self.completions_lm)

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

    async def get_score(self, generations: list, k=2):
        """
        Computes factscore for each generation

        topics: topics of the generations.
        generations: the generations we try to score.
        k: how many articles we want to add to the RAG context.
        """
        scores, decisions = [], []

        for gen in generations:

            facts, _ = await self.af_generator.run(gen) 

            gen_atoms = list(itertools.chain.from_iterable(facts.values()))

            if len(gen_atoms) == 0:
                scores.append(0)
                decisions.append(None)
                continue

            atoms_entities = await self.entities_retriever.run(gen_atoms)

            generation_decisions = await self.is_supported(atoms_entities, k=k)
    
            score = np.mean([d for d in generation_decisions.values()])

            decisions.append(generation_decisions)
            scores.append(score)

            out = {
                "decisions": decisions,
                "scores": scores,
            }

        return out

    async def is_supported(self, atoms_entities, k=2):
        """
        Maps `is_supported` boolean label to atoms
        """
        decisions = {}
        prompts = await self.get_rag_prompts_and_passages(atoms_entities, k)

        atoms = [p[0] for p in prompts]
        answers = await self.completions_lm.generate([p[1] for p in prompts])

        print(answers)

        for answer, atom in zip(answers, atoms):
            answer = answer.lower()

            if "true" in answer:
                is_supported = True
                decisions[atom] = is_supported
            if "false" in answer:
                is_supported = False
                decisions[atom] = is_supported
            else:
                continue
        
        print(decisions)

        return decisions


    async def get_rag_prompts_and_passages(self, atoms_entities, k: int) -> List[Tuple[str, str]]:
        """
        Search relevant titles, texts in db and Returns the retrieval part with appropriate info from wiki for each atomic fact
        """
        prompts = []
        atoms_entities_in_db = {}

        for atom, ents in atoms_entities.items():
            titles, texts = await self.db.search_text_by_queries(queries=ents, k=k)
            
            atoms_entities_in_db[atom] = titles 

            atom_passages = self.db.get_bm25_passages(fact=atom, texts=texts, n=k)

            rag_prompt = f"Task: Answer the question based on the given context.\n\n"
            for psg in reversed(atom_passages):
                context = f"Text: {psg}\n"
                rag_prompt += context

            rag_prompt += (
                    f'\n\Question: "{atom.strip()}" True or False? '
                    "Answer True if the information is supported by the context above and False otherwise.\n"
                    "Output: "
                )

            prompts.append((atom, rag_prompt))

        print(atoms_entities_in_db)
        return prompts


if __name__ == "__main__":
    fs = FactScorer()
    fs.register_knowledge_source(faiss_index="", data_db="", table_name="")
    print("DB registered!")

    gen1 = ["Albert Einstein was born on March 14, 1879, in the German city of Ulm beside the Danube River. His parents, Hermann Einstein and Pauline Koch, were middle-class secular Jews."]
    gen2 = ["During World War II, Alan Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence."]

    res = asyncio.run(fs.get_score(generations=gen2, k=2))

    print(res)
