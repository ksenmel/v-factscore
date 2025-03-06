import pickle
import faiss
import numpy as np
from tqdm import tqdm
from faiss.contrib.ondisk import merge_ondisk

'''
Faiss supports storing IVF indexes in a file on disk and accessing the file on-the-fly.
The simplest approach to do that -- on-disk index.
The on-disk index is built by merging the sharded indexes into one big index.
Useful links: 
https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM,
https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py
https://habr.com/ru/companies/okkamgroup/articles/509204/
'''

trained_index_name = ""
indexes_dir = ""
embeds_path = ""


def get_sharded_indexes(start: int, index_capacity: int, part: int, part_is_final=False,
                        batch_size=250):
    """
    computes embeddings to the titles with ids from <start> to <start + index_capacity> and loads them on the current index
    before using this function, you should already have trained IVF index from faiss, for example:
    index = faiss.index_factory(1536, "IVF32768,Flat") (IVF index with 32768 Voronoi cells and no quantization)

    s_start_idx: from what id to start adding vectors in the index
    part: number of the current idx
    part_is_final: if the current idx is final
    """
    with open(embeds_path, "rb") as f:
        titles = pickle.load(f)
    print(len(titles))

    index = faiss.read_index(indexes_dir + trained_index_name)
    print(index.ntotal)

    print("start adding")
    for i in tqdm(range(start, min(start + index_capacity, len(titles)), batch_size)):

        ids = [j for j in range(i, min(i + batch_size, len(titles)))]
        vecs = np.array([titles[idx] for idx in ids])
        if not part_is_final:
            assert len(vecs) == batch_size, f"batch size is {batch_size}, but got {len(vecs)} embeddings"
        vecs = np.array(vecs).astype(np.float16)
        index.add_with_ids(vecs, np.array(ids))
    faiss.write_index(index, indexes_dir + "block_%d.index" % part)
    print(index.ntotal)


def merge_sharded_indexes(number_of_indexes, final_index_name="all_vecs.index"):
    '''
    number_of_indexes: how many sharded indexes you have
    final_index_name: to what file the merged result will be saved
    '''
    print('loading trained index')
    index = faiss.read_index(indexes_dir + trained_index_name)
    block_fnames = [
        indexes_dir + "block_%d.index" % bno
        for bno in range(1, number_of_indexes)
    ]
    merge_ondisk(index, block_fnames, indexes_dir + "merged_index.ivfdata")
    print("write " + indexes_dir + final_index_name)
    faiss.write_index(index, indexes_dir + final_index_name)


if __name__ == '__main__':
    embeds_path = ""

    print("Stage 1")
    get_sharded_indexes(0, 250, 1, False)
    print("Stage 2")
    get_sharded_indexes(250, 500, 2, False)
    print("Stage 3")
    get_sharded_indexes(500, 750, 3, False)
    print("Stage 4")
    get_sharded_indexes(750, 1000, 4, True)

    merge_sharded_indexes(4)
