import asyncio
import os
import sys
import faiss
import sqlite3
import numpy as np
from factscore.api_requests import APIEmbeddingFunction

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class EmbedRetrieval:

    def __init__(self, index, data_db, ef: APIEmbeddingFunction):
        self.ef = ef
        self.index = faiss.read_index(index)
        self.connection = sqlite3.connect(data_db)

    async def search_top_k(self, query, k: int):
        """
        Find k titles with the closest embedding distance to the query.
        """
        assert isinstance(query, list)
        embed, _ = await self.ef(query)
        distances, indices = self.index.search(np.array(embed), k)
        distances, indices = distances[0], indices[0]
        # make it tuple
        texts = []
        titles = []

        print(indices)

        cursor = self.connection.cursor()

        for idx in indices:
            cursor.execute(f"SELECT title FROM documents WHERE rowid = {idx + 1}")
            id_results = cursor.fetchall()
            print(id_results)
            id_titles = id_results[0]
            titles.append(id_titles)
        cursor.close()
        return titles


if __name__ == "__main__":
    query = ["Albert Einstein"]

    llm = APIEmbeddingFunction(base_url="https://api.deepinfra.com/v1/openai/embeddings",
                               model_name="sentence-transformers/all-MiniLM-L12-v2",
                               dimensions=384)

    retrieval = EmbedRetrieval(index="",
                               data_db="",
                               ef=llm)

    result = asyncio.run(retrieval.search_top_k(query=query, k=5))

    titles = result
    print(titles)
