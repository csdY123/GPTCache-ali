from SSEM import SemanticSimilarity
import time

ssem = SemanticSimilarity(model_name='bert-base-multilingual-cased', metric='cosine',custom_embeddings=None)
output_sentences = ['This is a generated sentence 1.','This is a generated sentence 2.']
reference_sentences = ['This is the reference sentence 1.','This is the reference sentence 2.']
start_time = time.time()
similarity_score = ssem.evaluate(output_sentences, reference_sentences, n_jobs=1, level='sentence', output_format='mean')
print("Time consuming: {:.2f}s".format(time.time() - start_time))
print("Similarity score: ", similarity_score)