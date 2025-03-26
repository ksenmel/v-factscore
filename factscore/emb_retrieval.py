import asyncio
import os
import sys
import faiss
import sqlite3
import numpy as np

from factscore.api_requests import APIEmbeddingFunction


class EmbedRetrieval:

    def __init__(self, index, data_db, ef: APIEmbeddingFunction):
        self.ef = ef
        self.index = faiss.read_index(index)

        # need to move this to DocDB? (6M titles in RAM for each call not a good idea)
        self.connection = sqlite3.connect(data_db)
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT title FROM documents")
        self.titles = self.cursor.fetchall()

    async def search_top_k(self, queries, k: int):
        """
        Find k titles with the closest embedding distance to the query.
        """
        assert isinstance(queries, list)

        vecs, _ = await self.ef(queries)
        distances, indices = self.index.search(np.array(vecs), k)
        found_titles = []

        for query_idx, query_indices in enumerate(indices):
            for idx in query_indices:
                res = self.titles[idx]
                found_titles.append(res)
        return found_titles


if __name__ == "__main__":
    queries = ["Albert Einstein", "Alice in Wonderland"]

    llm = APIEmbeddingFunction(base_url="https://api.deepinfra.com/v1/openai/embeddings",
                               model_name="sentence-transformers/all-MiniLM-L12-v2",
                               dimensions=384)

    retrieval = EmbedRetrieval(index="",
                               data_db="",
                               ef=llm)

    result = asyncio.run(retrieval.search_top_k(queries=queries, k=2))

    titles = result
    print(titles)
