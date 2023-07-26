import os
from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_faiss

import_faiss()

import faiss  # pylint: disable=C0413


class Faiss(VectorBase):
    """vector store: Faiss
Faiss（Facebook AI Similarity Search）是一个高性能的相似性搜索库，用于在大规模数据集中快速搜索最相似的向量。
    :param index_path: the path to Faiss index, defaults to 'faiss.index'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    """

    def __init__(self, index_file_path, dimension, top_k):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._index = faiss.index_factory(self._dimension, "IDMap,Flat", faiss.METRIC_L2)
        self._top_k = top_k
        #print("top_k:",top_k)
        if os.path.isfile(index_file_path):
            self._index = faiss.read_index(index_file_path)


    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        #data_array：list:1 (786,)
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)    #[4]
        self._index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray, top_k: int = -1):
        if self._index.ntotal == 0:
            return None
        if top_k == -1:
            top_k = self._top_k
        np_data = np.array(data).astype("float32").reshape(1, -1)   #.reshape(1, -1) 调整数组的形状为 (1, -1)，其中 -1 表示自动计算数组的大小，以保持数组中元素的总数不变。这样的形状调整通常用于将一维数组转换为二维数组。
        dist, ids = self._index.search(np_data, top_k)  #self._index 是 Faiss 索引对象，可能是一个已经构建好的 Faiss 索引，用于存储和加速向量数据的搜索。
        ids = [int(i) for i in ids[0]]
        return list(zip(dist[0], ids))

    def rebuild(self, ids=None):
        return True

    def delete(self, ids):
        ids_to_remove = np.array(ids).astype('int64')
        self._index.remove_ids(faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove)))

    def flush(self):
        faiss.write_index(self._index, self._index_file_path)

    def close(self):
        self.flush()

    def count(self):
        return self._index.ntotal
