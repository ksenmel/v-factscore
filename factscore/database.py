import asyncio
import sqlite3
import numpy as np

from rank_bm25 import BM25Okapi

from factscore.api_requests import APIEmbeddingFunction
from factscore.emb_retrieval import EmbedRetrieval


class DocDB:
    """Sqlite document storage. By default, uses Wikipedia dump from 2023/04/01."""
    def __init__(self, db_path: str, fais_index: str, ef: APIEmbeddingFunction, table: str):
        self.connection = sqlite3.connect(db_path)
        self.ef = ef

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            if db_path is not None:
                Exception(f"{db_path} is empty")

        cursor.execute(f"SELECT title FROM {table}")
        self.titles = [row[0] for row in cursor.fetchall()]

        self.retriever = EmbedRetrieval(index=fais_index, ef=ef, titles=self.titles, connection=self.connection)

    async def search_text_by_queries(self, queries, k):
        res = await self.retriever.search(queries=queries, k=k)
        return res

    async def postprocess_text(self):
        pass

    def get_bm25_passages(self, topic, question, texts, k):
        """
        Returns k passages (parts of the texts) most similar to the topic using bm25
        """
        query = topic + " " + question.strip() if topic is not None else question.strip()
        bm25 = BM25Okapi([text["text"].replace("<s>", "").replace(
            "</s>", "").split() for text in texts])
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [texts[i] for i in indices]


if __name__ == "__main__":
    queries = ["Albert Einstein was born in 14 May 1981"]
    ef = APIEmbeddingFunction(base_url="https://api.deepinfra.com/v1/openai/embeddings",
                              model_name="sentence-transformers/all-MiniLM-L6-v2",
                              dimensions=384)

    db = DocDB(db_path="", fais_index="", ef=ef, table='')

    result = asyncio.run(db.search_text_by_queries(queries=queries, k=5))

    titles, texts = result
    print(titles)
    # print(texts)