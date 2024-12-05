from documentEmbedding import DocumentEmbedding

d = DocumentEmbedding("../../datasets/full_docs_small")
d.pretrain_dataset(True)



# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#
# file = open("../datasets/full_docs_small/output_1.txt")
# text = file.read()
# text
#
#
# tokenized_text = tokenizer(text, truncation=True, return_tensors="pt")
# print(tokenized_text)




# from sentence_transformers import SentenceTransformer
#
# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]
#
# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# # [3, 384]
#
# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# # tensor([[1.0000, 0.6660, 0.1046],
# #         [0.6660, 1.0000, 0.1411],
# #         [0.1046, 0.1411, 1.0000]])