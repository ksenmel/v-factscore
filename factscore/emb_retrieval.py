import faiss
import numpy as np

from sqlite3 import Connection
from factscore.api_requests import APIEmbeddingFunction


class EmbedRetrieval:
    def __init__(
        self, index, titles: list[str], ef: APIEmbeddingFunction, connection: Connection
    ):
        self.ef = ef
        self.index = faiss.read_index(index)
        self.index.nprobe = 16
        self.connection = connection
        self.titles = titles

    async def search(self, queries, k: int):
        """
        Find k titles with the closest embedding distance to the query.
        """
        assert isinstance(queries, list)

        vecs, _ = await self.ef(queries)
        distances, indices = self.index.search(np.array(vecs), k)
        k_titles = []
        k_texts = []

        cursor = self.connection.cursor()
        for query, ids in enumerate(indices):
            for idx in ids:
                title = self.titles[idx]

                # TODO: replace documents with argument 'table'
                cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
                text = [i[0] for i in cursor.fetchall()]

                k_titles.append(title)
                k_texts.append(text)

        return k_titles, k_texts
