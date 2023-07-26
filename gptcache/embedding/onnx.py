import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_huggingface()
import_onnxruntime()
import_huggingface_hub()

from transformers import AutoTokenizer, AutoConfig  # pylint: disable=C0413
from huggingface_hub import hf_hub_download  # pylint: disable=C0413
import onnxruntime  # pylint: disable=C0413

#https://chat.openai.com/share/d7641858-4baa-4318-974c-e3cb454c722a
class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.
        使用 ONNX 模型为给定文本生成文本嵌入。
    Example:
        .. code-block:: python

            from gptcache.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model='GPTCache/paraphrase-albert-onnx')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="GPTCache/paraphrase-albert-onnx"):
        tokenizer_name = "GPTCache/paraphrase-albert-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  #表示使用预训练模型的名称（tokenizer_name）来加载一个预训练的tokenizer
        self.model = model
        #C:\Users\LENOVO\.cache\huggingface\hub\models--GPTCache--paraphrase-albert-onnx\snapshots\5b562a100bc67e898ac89814e7a4668a18d65756
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx") #根据提供的参数从Hugging Face模型仓库下载模型，并将模型保存到指定的本地路径。
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)    #通过这个会话对象，你可以执行模型的推理操作，输入数据并获取模型的输出结果
        config = AutoConfig.from_pretrained(
            "GPTCache/paraphrase-albert-small-v2"
        )   #这段代码使用 Hugging Face Transformers 库的 AutoConfig.from_pretrained() 方法，从预训练模型库中加载指定模型的配置文件。
        self.__dimension = config.hidden_size   #768

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string. string类文本，原问题
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        #tokenizer的作用：将原始文本转换为模型可以理解的离散单元序列。这样处理后的文本数据才能被输入到模型中进行文本嵌入的计算和训练。
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")
        #encoded_text 是一个包含了文本编码结果的字典对象，它包含了分词器对文本进行编码后得到的 input_ids、attention_mask 和 token_type_ids 等信息。
        #综上所述，这段代码的作用是将文本编码结果转换为适合输入 ONNX 模型的数据格式，并存储在 ort_inputs 字典中，以便后续使用这些输入进行模型推理。
        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).astype("int64").reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).astype("int64").reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).astype("int64").reshape(1, -1),
        }
        #执行 ONNX 模型的推理，将 ort_inputs 作为输入传递给模型，并返回模型的输出结果
        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()    #调用这个方法可以将数组展平为一维数组

    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        #将每个位置的特征向量加权和除以对应位置的注意力掩码和，得到句子级别的表示向量
        #计算得到句子级别的表示向量，并返回该向量作为最终的处理结果
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
