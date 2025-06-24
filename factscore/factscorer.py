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


if __name__ == "__main__":
    fs = FactScorer()
    fs.register_knowledge_source(
        faiss_index="/Users/kseniia/factscore/indexes/all_vecs.index",
        data_db="/Users/kseniia/enwiki-20230401.db",
        table_name="documents",
    )
    print("DB registered!")

    gen1 = "Albert Einstein (1879–1955) was a German-born theoretical physicist who revolutionized modern science with his groundbreaking theories, most notably the theory of relativity. Born on March 14, 1879, in Ulm, Germany, Einstein showed an early curiosity for mathematics and physics, though he struggled with the rigid structure of traditional schooling. In 1915, he expanded his work into General Relativity, redefining gravity as the curvature of spacetime by mass and energy. His predictions, such as light bending around the sun, were confirmed during the 1919 solar eclipse, making him a global celebrity."
    gen3 = "Jessie Mae Brown Beavers (born 1908, date of death unknown) was an African-American social worker, community leader, and activist who played a significant role in promoting civil rights and social justice in Los Angeles, California, during the mid-20th century.\n\nBorn in Texas, Beavers moved to Los Angeles in the 1920s. She earned her bachelor's degree in social work from the University of California, Los Angeles (UCLA) and later pursued her master's degree from the University of Southern California (USC).\n\nJessie Mae Brown Beavers was a dedicated social worker who spent her career advocating for the well-being of the African-American community in Los Angeles. She held several positions in social work agencies, including the Los Angeles County Department of Charities, where she worked to improve the conditions of African-American wards in juvenile detention centers. She was also a member of the National Association of Social Workers (NASW).\n\nIn addition to her work as a social worker, Beavers was an active member of several civil rights organizations, such as the National Association for the Advancement of Colored People (NAACP) and the Urban League. She was instrumental in organizing and leading community-based programs aimed at improving education, employment, and housing opportunities for African-Americans.\n\nBeavers was also a prominent member of the Alpha Kappa Alpha (AKA) sorority, the first African-American Greek-lettered sorority, and held leadership positions within the organization. Her efforts in the AKA sorority helped to strengthen its commitment to social justice, community service, and the promotion of African-American culture.\n\nThroughout her life, Jessie Mae Brown Beavers remained a tireless advocate for racial equality and social justice. Although the exact date of her death is unknown, her legacy lives on through her many contributions to the improvement of the African-American community in Los Angeles and beyond."
    gen2 = """
    Tomography is imaging by sections or sectioning that uses any kind of penetrating wave. The method is used in radiology, archaeology, biology, atmospheric science, geophysics, oceanography, plasma physics, materials science, cosmochemistry, astrophysics, quantum information, and other areas of science. 
    In many cases, the production of these images is based on the mathematical procedure tomographic reconstruction, such as X-ray computed tomography technically being produced from multiple projectional radiographs. Many different reconstruction algorithms exist. Most algorithms fall into one of two categories: filtered back projection (FBP) and iterative reconstruction (IR). These procedures give inexact results: they represent a compromise between accuracy and computation time required. FBP demands fewer computational resources, while IR generally produces fewer artifacts (errors in the reconstruction) at a higher computing cost.
    """
    
    gen4 = "There are several individuals named Joseph A. Lopez, but without any specific context or background information, it is difficult to provide a biography. Please provide more details, such as occupation or field of expertise, to help me identify the correct Joseph A. Lopez you are referring to."
    gen5 = "Maxime Masson is a fairly common name, and there might be multiple individuals with that name. As an AI language model, I cannot provide a biography of a specific individual named Maxime Masson without more specific information.\n\nIf you're referring to a public figure or someone well-known, kindly provide more context or details related to their profession, accomplishments, or field of expertise to help me generate a relevant and accurate biography."
    gen6 = "I'm sorry, but I cannot find any information on a person named Serena Tideman. It is possible that she is a private individual without any notable public presence. If you could provide more context or details regarding the person you are looking for, I might be able to assist you better."

    res = asyncio.run(fs.get_score(generations=[gen6], k=2))

    print(res)
