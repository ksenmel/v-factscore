import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

from factscore.api_requests import APIEmbeddingFunction
from factscore.emb_retrieval import EmbedRetrieval

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class DocDB:
    """Sqlite document storage. By default, uses Wikipedia dump from 2023/04/01."""

    def __init__(
        self, db_path: str, faiss_index: str, ef: APIEmbeddingFunction, table: str
    ):
        self.connection = sqlite3.connect(db_path)
        self.ef = ef

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            if db_path is not None:
                Exception(f"{db_path} is empty")

        cursor.execute(f"SELECT title FROM {table}")
        self.titles = [row[0] for row in cursor.fetchall()]

        self.retriever = EmbedRetrieval(
            index=faiss_index, ef=ef, titles=self.titles, connection=self.connection
        )

    async def search_text_by_queries(self, queries, k):
        res = await self.retriever.search(queries=queries, k=k)
        return res

    def get_bm25_passages(self, topic, question, text, k):
        """
        Returns k passages most similar to the topic + question using BM25
        """
        #  .replace("</s>", "")?
        texts = text[0].split("</s>")
        print(texts)
        query = f"{topic} {question.strip()}" if topic else question.strip()
        bm25 = BM25Okapi(texts)
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [texts[i] for i in indices]


def postprocess_text(texts: list[list[str]]):
    """
    Needed to process text before using BM25 and RAG
    """
    for sublist in texts:
        for i in range(len(sublist)):
            sublist[i] = (
                sublist[i].replace("<s>", "").replace(f"{SPECIAL_SEPARATOR}", "")
            )
