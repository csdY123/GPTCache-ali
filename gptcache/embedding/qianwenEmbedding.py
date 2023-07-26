import importlib
from typing import Optional
import numpy as np
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import prompt_install
class QianWenEmbedding(BaseEmbedding):
    def __init__(self, model: str = "en", dim: int = None):
        # def _check_library(libname: str, prompt: bool = True, package: Optional[str] = None):
        #     is_avail = False
        #     if importlib.util.find_spec(libname):
        #         is_avail = True
        #     if not is_avail and prompt:
        #         prompt_install(package if package else libname)
        #     return is_avail
        # _check_library("sentence-transformers")
        self.embeddingModel = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.__dimension = len(self.embeddingModel.embed_query("foo"))
        print("self.__dimension:",self.__dimension)
    def to_embeddings(self, data, **_):
        assert isinstance(data, str), "Only allow string as input."
        emb = self.embeddingModel.embed_query(data)
        return np.array(emb).astype("float32")
    @property
    def dimension(self):
        return self.__dimension
# qianWenEmbedding=QianWenEmbedding()
# print(qianWenEmbedding.dimension)