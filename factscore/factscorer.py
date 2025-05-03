import asyncio
import itertools
import numpy as np
import time

from rank_bm25 import BM25Okapi
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
        scores, decisions = [], []

        for gen in generations:

            start_time = time.time()
            facts, _ = await self.af_generator.run(gen) 
            end_time = time.time()

            print(f"Generated atomics for {end_time - start_time} seconds")

            gen_atoms = list(itertools.chain.from_iterable(facts.values()))

            start_time = time.time()
            atoms_entities = await self.entities_retriever.run(gen_atoms)
            end_time = time.time()

            print(f"Retrieved entities for {end_time - start_time} seconds")

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

        print(corpus)

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

    gen1 = "Albert Einstein was born on March 14, 1879, in the German city of Ulm beside the Danube River. His parents, Hermann Einstein and Pauline Koch, were middle-class secular Jews."
    gen2 = "During World War II, Alan Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence."
    gen3 = "Elvis Presley (1935–1977) was an American singer and actor widely regarded as the 'King of Rock and Roll'. "

    gen4 = "Elvis Presley (1935–1977) was an American singer and actor widely regarded as the “King of Rock and Roll.” Elvis Presley is one of the most significant cultural icons of the 20th century and helped revolutionize popular music with his energetic performance style, powerful voice, and charismatic stage presence. Born on January 8, 1935, in Tupelo, Mississippi, Elvis Presley moved to Memphis, Tennessee, as a teenager. In 1954, Elvis Presley began his music career with Sun Records, blending country, blues, and gospel influences into a new sound that would come to be known as rock and roll. His breakout hit, “Heartbreak Hotel,” in 1956 launched him to national fame. In addition to music, Elvis Presley had a successful film career, starring in over 30 movies throughout the 1960s. After a brief break from live performances, Elvis Presley made a legendary comeback with a 1968 TV special and went on to perform sold-out shows in Las Vegas during the 1970s. Despite his success, Elvis Presley struggled with health problems and prescription drug dependence. Elvis Presley died of a heart attack on August 16, 1977, at his home, Graceland, in Memphis, at age 42."

    start_time = time.time()
    res = asyncio.run(fs.get_score(generations=[gen4], k=2))
    end_time = time.time()

    print(res)
    print(f"Processed whole program for {end_time - start_time} seconds")
