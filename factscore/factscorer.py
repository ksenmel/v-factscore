import asyncio
import itertools
import numpy as np
import time

from rank_bm25 import BM25Okapi
from typing import List, Tuple

from factscore.api_requests import APICompletions
# from factscore.atomic_facts import AtomicFactGenerator
from factscore.atomic_facts import GenerationAtomicFactGenerator
from factscore.api_requests import APIEmbeddingFunction
from factscore.database import DocDB
from factscore.entities_retriever import EntitiesRetriever


class FactScorer:
    def __init__(
        self,
        completions_base_url="http://localhost:11434/api/chat",
        completions_model_name="gemma3:1b",
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
        results = {
                "decisions": [],
                "scores": [],
                "process_time": []
            }
        
        max_retries = 3
        for gen in generations:
            attempt = 0
            success = False
            while attempt < max_retries and not success:
                try: 
                    start_time = time.time()

                    facts = await self.af_generator.run(gen) 

                    gen_atoms = list(itertools.chain.from_iterable(facts.values()))

                    atoms_entities = await self.entities_retriever.run(gen_atoms)
                    
                    gen_decisions = await self.is_supported(atoms_entities, k=k)

                    end_time = time.time()
            
                    score = round(np.mean([d for d in gen_decisions.values()]), 4)
                    gen_process_time = round(end_time - start_time, 4)

                    results["decisions"].append(gen_decisions)
                    results["scores"].append(score)
                    results["process_time"].append(gen_process_time)

                    success = True
                    print(results)

                    return results

                except Exception as e:
                    attempt += 1
                    print(f"ERROR {e} for generation {gen}. Attempt {attempt}/{max_retries}")
                    
                    if attempt < max_retries:
                        await asyncio.sleep(3)
                    else:
                        results["decisions"].append(None)
                        results["scores"].append(0)
                        results["process_time"].append(0)
                    pass

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

        return decisions


    async def get_rag_prompts_and_passages(self, atoms_entities, k: int) -> List[Tuple[str, str]]:
        """
        Search relevant titles, texts in db and Returns the retrieval part with appropriate info from wiki for each atomic fact
        """
        prompts = []

        ents = list(set(itertools.chain.from_iterable(atoms_entities.values())))

        start_time = time.time()
        _, texts = await self.db.search_text_by_queries(queries=ents, k=k)
        end_time = time.time()

        print(f"Searched texts with FAISS in {end_time - start_time} seconds")

        start_time = time.time()
        corpus = []
        for text in texts:
            sents = self.segmenter.segment(text[0])
            corpus.extend(sents)

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        for atom in atoms_entities.keys():

            atom_passages = bm25.get_top_n(atom.split(" "), corpus, n=k)

            rag_prompt = f"Task: Answer the question based on the given context.\n\n"

            for psg in reversed(atom_passages):
                context = f"Text: {psg}\n"
                rag_prompt += context

            rag_prompt += (f'\n\Question: "{atom.strip()}" True or False? '
                            "Answer True if the information is supported by the context above and False otherwise.\n"
                            "Output: "
                            )
            
            prompts.append((atom, rag_prompt))

        end_time = time.time()

        print(f"Did BM25 in {end_time - start_time} seconds")

        return prompts


if __name__ == "__main__":
    fs = FactScorer()
    fs.register_knowledge_source(faiss_index="/Users/kseniia/factscore/indexes/all_vecs.index", data_db="/Users/kseniia/enwiki-20230401.db", table_name="documents")
    print("DB registered!")

    gen1 = "Jessie Mae Brown Beavers (born 1908, date of death unknown) was an African-American social worker, community leader, and activist who played a significant role in promoting civil rights and social justice in Los Angeles, California, during the mid-20th century. She earned her bachelor's degree in social work from the University of California, Los Angeles (UCLA) and later pursued her master's degree from the University of Southern California (USC)."
    gen2 = "Elvis Presley (1935–1977) was an American singer and actor, widely celebrated as the 'King of Rock and Roll'. He became one of the most influential figures in 20th-century popular culture through his music, charisma, and groundbreaking performance style. Born on January 8, 1935, in Tupelo, Mississippi, Elvis grew up in a modest household with a strong influence from gospel music. His family later moved to Memphis, Tennessee, where he was exposed to a blend of blues, country, and rhythm & blues — elements that would shape his unique musical style."
    res = asyncio.run(fs.get_score(generations=[gen1, gen2], k=2))

    print(res['scores'])
    print(res['process_time'])
