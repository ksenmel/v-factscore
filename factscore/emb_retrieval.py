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

        # need to move this to DocDB?
        self.connection = sqlite3.connect(data_db)
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT title FROM documents")
        self.titles = [row[0] for row in self.cursor.fetchall()]

    async def search(self, queries, k: int):
        """
        Find k titles with the closest embedding distance to the query.
        """
        assert isinstance(queries, list)

        vecs, _ = await self.ef(queries)
        distances, indices = self.index.search(np.array(vecs), k)
        k_titles = []
        k_texts = []

        for query, ids in enumerate(indices):
            for idx in ids:
                title = self.titles[idx]

                self.cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
                text = [i[0] for i in self.cursor.fetchall()]

                k_titles.append(title)
                k_texts.append(text)

        return k_titles, k_texts



if __name__ == "__main__":
    queries = ["Albert Einstein"]

    llm = APIEmbeddingFunction(base_url="https://api.deepinfra.com/v1/openai/embeddings",
                               model_name="sentence-transformers/all-MiniLM-L6-v2",
                               dimensions=384)

    retrieval = EmbedRetrieval(index="ivf.index",
                               data_db="/Users/kseniia/enwiki-20230401.db",
                               ef=llm)

    result = asyncio.run(retrieval.search(queries=queries, k=5))

    titles, texts = result
    print(titles)
    # print(texts)
