from documentEmbedding import DocumentEmbedding
from invertedIndex import InvertedIndex
from queryProcessing import QueryProcessing
from testQueries import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
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

documentEmbedding = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_mean", True)
documentEmbedding.pretrain_dataset(False)


queryProcessing = QueryProcessing(documentEmbedding)

# print("start query")
# time_start = time.perf_counter()
# #print(queryProcessing.processQueryLoop("how much is a cost to run disneyland", 10))
# print(queryProcessing.processQuery("how much is a cost to run disneyland", 10))
# time_elapsed = (time.perf_counter() - time_start)
# print(time_elapsed)
#
# print(evaluation(queryProcessing))

# time_start = time.perf_counter()
# test_queries(queryProcessing, True)
# time_elapsed = (time.perf_counter() - time_start)
# print(time_elapsed)
#
# print("trained")
invertedIndex = InvertedIndex(documentEmbedding)
#
invertedIndex.kMeansCluster(50)
#
# time_start = time.perf_counter()
# test_queries(invertedIndex, False)
# time_elapsed = (time.perf_counter() - time_start)
# print(time_elapsed)
print(evaluation(invertedIndex, True))

# print("search")
# # 5. Query processing: embed the query and find the closest cluster
# query = "how much is a cost to run disneyland"
# query_embedding = model.encode([query])
# print("start query")
# time_start = time.perf_counter()
# print("Relevant Documents: ", invertedIndex.getDocuments(query_embedding, 10, 10))
# time_elapsed = (time.perf_counter() - time_start)
# print(time_elapsed)











# # GPT
# similarities = cosine_similarity(query_embedding, invertedIndex.centroids)
# print(similarities)
# closest_cluster = np.argmax(similarities)
# print(closest_cluster)
# print(len(invertedIndex.inverted_index))
# # Retrieve relevant documents from the closest cluster
# relevant_docs = invertedIndex.inverted_index[closest_cluster]
#
# # Print relevant document indices
# print("Relevant Documents: ", relevant_docs)
