import asyncio
import itertools
import numpy as np
import time

from rank_bm25 import BM25Okapi
from typing import List, Tuple

from factscore.api_requests import APICompletions
from factscore.atomic_facts import GenerationAtomicFactGenerator
from factscore.api_requests import APIEmbeddingFunction
from factscore.database import DocDB
from factscore.entities_retriever import EntitiesRetriever


class FactScorer:
    def __init__(
        self,
        completions_base_url="https://api.deepinfra.com/v1/openai/chat/completions",
        completions_model_name="Qwen/Qwen3-32B",
        embedding_base_url="http://localhost:11434/api/embeddings",
        embedding_model_name="nomic-embed-text",
    ):
        self.completions_lm = APICompletions(
            base_url=completions_base_url, model_name=completions_model_name
        )
        self.embeddings_lm = APIEmbeddingFunction(
            base_url=embedding_base_url, model_name=embedding_model_name
        )
        self.af_generator = GenerationAtomicFactGenerator(llm=self.completions_lm)
        self.entities_retriever = EntitiesRetriever(llm=self.completions_lm)

        self.segmenter = self.af_generator.segmenter

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
        results = {"decisions": [], "scores": [], "process_time": [], "estimated_costs": []}

        max_retries = 3
        retry_delay = 3

        for gen in generations:
            attempt = 0
            success = False

            while attempt < max_retries and not success:
                try:
                    start_time = time.time()

                    facts, facts_cost = await self.af_generator.run(gen)
                    
                    gen_atoms = list(itertools.chain.from_iterable(facts.values()))

                    if gen_atoms:
                        atoms_entities, ents_cost = await self.entities_retriever.run(gen_atoms)

                        gen_decisions, decisions_cost = await self.is_supported(atoms_entities, k=k)

                        end_time = time.time()

                        score = round(np.mean([d for d in gen_decisions.values()]), 4)
                        gen_process_time = round(end_time - start_time, 4)
                        cost = facts_cost + ents_cost + decisions_cost

                        results["decisions"].append(gen_decisions)
                        results["scores"].append(score)
                        results["process_time"].append(gen_process_time)
                        results["estimated_costs"].append(cost)

                    else:
                        results["decisions"].append(None)
                        results["scores"].append(0)
                        results["process_time"].append(0)
                        results["estimated_costs"].append(0.0)

                    success = True
                    
                except Exception as e:
                    attempt += 1

                    error_msg = (
                        f"ERROR {e} for generation {gen}.\n"
                        f"Attempt {attempt}/{max_retries}."
                    )
                    print(error_msg)

                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)

            if not success:
                print(f"Failed to process generation {gen} after {max_retries} attempts.")
                
                results["decisions"].append(None)
                results["scores"].append(0)
                results["process_time"].append(0)
                results["estimated_costs"].append(0.0)
            
            print(gen_decisions)
            print(f"gen factscore: {score}")
            print(f"gen pocess time: {gen_process_time}")
            print(f"gen estimated cost: {cost}")
        
        return results

    async def is_supported(self, atoms_entities, k=2):
        """
        Maps `is_supported` boolean label to atoms
        """
        decisions = {}
        prompts = await self.get_rag_prompts_and_passages(atoms_entities, k)

        atoms = [p[0] for p in prompts]

        answers, failed, cost = await self.completions_lm.generate([p[1] for p in prompts])

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

        return decisions, cost

    async def get_rag_prompts_and_passages(
        self, atoms_entities, k: int
    ) -> List[Tuple[str, str]]:
        """
        Search relevant titles, texts in db and Returns the retrieval part with appropriate info from wiki for each atomic fact
        """
        prompts = []

        ents = list(set(itertools.chain.from_iterable(atoms_entities.values())))

        titles, texts = await self.db.search_text_by_queries(queries=ents, k=k)

        corpus = []
        for text in texts:
            sents = self.segmenter.segment(text[0])
            corpus.extend(sents)

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        for atom in atoms_entities.keys():
            atom_passages = bm25.get_top_n(atom.split(" "), corpus, 5)
            
            rag_prompt = f"Task: Answer the question based on the given context.\n\n"

            for psg in atom_passages:
                context = f"Text: {psg}\n"
                rag_prompt += context

            rag_prompt += (
                f'\n\Question: "{atom.strip()}" True or False? '
                "Answer True if the information is supported by the context above and False otherwise.\n"
                "Output: "
            )

            prompts.append((atom, rag_prompt))

        return prompts
