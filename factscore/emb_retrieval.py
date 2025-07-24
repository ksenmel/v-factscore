import faiss
import numpy as np

from sqlite3 import Connection
from factscore.api_requests import APIEmbeddingFunction


class EmbedRetrieval:
    """
    A class for performing embedding-based retrieval using FAISS index and SQL database.

    This class enables efficient similarity search by:
    1. Converting text queries to embeddings
    2. Searching a FAISS index for nearest neighbors
    3. Retrieving corresponding documents from a SQL database

    Attributes:
        ef (APIEmbeddingFunction): Embedding function for query encoding
        index (faiss.Index): FAISS index for similarity search
        connection (sqlite3.Connection): Database connection for document retrieval
        titles (List[str]): List of document titles corresponding to FAISS index
    """

    def __init__(
        self, index, titles: list[str], ef: APIEmbeddingFunction, connection: Connection
    ):
        self.ef = ef
        self.index = faiss.read_index(index)
        """
        Setting nprobe (default nprobe is 1) defines how many nearby Voroni cells to search.

        The nprobe parameter is always a way of adjusting the tradeoff between speed and accuracy 
        of the result. Setting nprobe = nlist gives the same result as the brute-force search (but slower)
        """
        self.index.nprobe = 8

        self.connection = connection
        self.titles = titles

    async def search(self, queries, k: int):
        """
        Find k titles with the closest embedding distance to the query
        """
        assert isinstance(queries, list)

        vecs, failed, cost = await self.ef(queries)

        vecs = np.array(vecs)
        _, indices = self.index.search(vecs, k)
        k_titles = []
        k_texts = []

        cursor = self.connection.cursor()
        for query, ids in enumerate(indices):
            for idx in ids:
                title = self.titles[idx]

                cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
                text = [i[0] for i in cursor.fetchall()]

                k_titles.append(title)
                k_texts.append(text)

        return k_titles, k_texts
