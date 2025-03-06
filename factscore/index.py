import pickle
import faiss
import numpy as np
from tqdm import tqdm

'''
Faiss supports storing IVF indexes in a file on disk and accessing the file on-the-fly.
The simplest approach to do that -- on-disk index.
The on-disk index is built by merging the sharded indexes into one big index.
Useful links: 
https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM,
https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py
'''


def get_sharded_indexes(s_start_idx: int, index_capacity: int, trained_index_path: str,
                        indexes_dir: str, part: int, embeds_file_path: str, part_is_final=False, batch_size=250):
    """
    computes embeddings to the titles with ids from <start> to <start + index_capacity> and loads them on the current index
    before using this function, you should already have trained IVF index from faiss, for example:
    index = faiss.index_factory(1536, "IVF32768,Flat") (IVF index with 32768 Voronoi cells and no quantization)

    s_start_idx: from what id to start adding vectors in the index
    part: number of the current idx
    part_is_final: if the current idx is final
    """
    index = faiss.read_index(trained_index_path)

    with open(embeds_file_path, "rb") as f:
        titles_emb = pickle.load(f)

    s_end_idx = min(s_start_idx + index_capacity, len(titles_emb) - 1)

    for i in tqdm(range(s_start_idx, s_end_idx, batch_size)):
        ids = [j for j in range(i, min(i + batch_size, s_end_idx))]
        # titles_to_add = titles_emb[ids[0]: ids[-1] + 1]
        titles_to_add = np.array([titles_emb[idx] for idx in ids])
        if not part_is_final:
            assert len(titles_to_add) == batch_size, f"batch size is {batch_size}, but got {len(titles_to_add)} embeddings"
        index.add_with_ids(titles_to_add, ids)
    faiss.write_index(index, indexes_dir + "/" + "block_%d.index" % part)


if __name__ == '__main__':
    index_path = "titles-ivf-1000000.index"
    embeds_path = "wiki_embeddings-1000.pkl"

    print("Stage 1")
    get_sharded_indexes(0, 250, index_path, "indexes", 1, embeds_path, False)
    print("Stage 2")
    get_sharded_indexes(249, 500, index_path, "indexes", 2, embeds_path, False)
    print("Stage 3")
    get_sharded_indexes(499, 750, index_path, "indexes", 3, embeds_path, False)
    print("Stage 4")
    get_sharded_indexes(749, 1000, index_path, "indexes", 4, embeds_path, True)

