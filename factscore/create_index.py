import faiss
import numpy as np
import sqlite3

from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm

from factscore.api_requests import APIEmbeddingFunction

"""
Faiss supports storing IVF indexes in a file on disk and accessing the file on-the-fly.
The on-disk index is built by merging the sharded indexes into one big index.

https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM,
"""

trained_index_path = ""
index_capacity = 500000
indexes_dir = ""
db_path = ""

connection = sqlite3.connect(db_path)
cursor = connection.cursor()
results = cursor.execute(f"SELECT title FROM documents")
titles = results.fetchall()

ef = APIEmbeddingFunction(base_url="", model_name="")


async def get_sharded_indexes(
    start: int, index_capacity: int, part: int, part_is_final=False, batch_size=1000
):
    """
    Computes embeddings to the titles with ids from <start> to <start + index_capacity> and loads them on the sharded index.
    Before using this function, you should already have trained IVF index from faiss

    :param start: from what id to start adding vectors in the index
    :param index_capacity: how many indexes will be in the shard
    :param part: number of the current shard part
    :param part_is_final: if the current shard is final
    :param batch_size: the size of the batch
    """
    print(f"Stage {part}")

    index = faiss.read_index(trained_index_path)

    print("start adding")
    for i in tqdm(range(start, min(start + index_capacity, len(titles)), batch_size)):
        ids = [j for j in range(i, min(i + batch_size, len(titles)))]
        titles_to_add = list(map(lambda x: str(x[0]), titles[ids[0] : ids[-1] + 1]))
        vecs, _ = await ef(titles_to_add)

        if not part_is_final:
            assert (
                len(vecs) == batch_size
            ), f"batch size is {batch_size}, but got {len(vecs)} embeddings"

        vecs = np.array(vecs).astype(np.float32)
        index.add_with_ids(vecs, np.array(ids))
    faiss.write_index(index, indexes_dir + "block_%d.index" % part)


def merge_sharded_indexes(number_of_indexes, final_index_name="all_vecs.index"):
    """
    :param number_of_indexes: how many sharded indexes you have
    :param final_index_name: to what file the merged result will be saved
    """
    print("loading trained index")
    index = faiss.read_index(trained_index_path)
    block_fnames = [
        indexes_dir + "block_%d.index" % bno for bno in range(1, number_of_indexes)
    ]
    merge_ondisk(index, block_fnames, indexes_dir + "merged_index.ivfdata")
    print("write " + indexes_dir + final_index_name)
    faiss.write_index(index, indexes_dir + final_index_name)
