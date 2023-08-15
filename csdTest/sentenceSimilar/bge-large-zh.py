from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers.util import cos_sim
#语义相似性的一个测试
# Sentences we want sentence embeddings for
sentences = ["你知道北京吗", "你知道北京吗"]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh',cache_dir="..\csdTest")
model = AutoModel.from_pretrained('BAAI/bge-large-zh',cache_dir="..\csdTest")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for retrieval task, add an instruction to query
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
print(cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item())
