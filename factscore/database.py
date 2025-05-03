import sqlite3

from factscore.api_requests import APIEmbeddingFunction
from factscore.emb_retrieval import EmbedRetrieval


class DocDB:
    """Sqlite document storage. By default, uses Wikipedia dump from 2023/04/01"""

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
