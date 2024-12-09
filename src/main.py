from documentEmbedding import DocumentEmbedding
from invertedIndex import InvertedIndex
from queryProcessing import QueryProcessing
from testQueries import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

###FIRST PART###
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# documentEmbedding = DocumentEmbedding("../../datasets/full_docs_small", model)
# documentEmbedding.pretrain_dataset(False)
#
# queryProcessing = QueryProcessing(documentEmbedding)
#
# test_queries(queryProcessing,True)
# print(evaluation(queryProcessing,True))

###SECOND PART###
model = SentenceTransformer("all-MiniLM-L6-v2")

documentEmbedding = DocumentEmbedding("../../datasets/full_docs_small", model)
documentEmbedding.pretrain_dataset(False)

invertedIndex = InvertedIndex(documentEmbedding)

invertedIndex.kMeansCluster(5)

# 5. Query processing: embed the query and find the closest cluster
query = "what agency can i report a scammer concerning my computer"
query_embedding = model.encode([query])


# GPT
# similarities = cosine_similarity(query_embedding, invertedIndex.centroids)
# closest_cluster = np.argmax(similarities)
#
# # Retrieve relevant documents from the closest cluster
# relevant_docs = invertedIndex.inverted_index[closest_cluster]
#
# # Print relevant document indices
# print("Relevant Documents: ", relevant_docs)