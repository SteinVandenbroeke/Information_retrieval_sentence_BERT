from documentEmbedding import DocumentEmbedding
from invertedIndex import InvertedIndex
from queryProcessing import QueryProcessing
from testQueries import *
from sentence_transformers import SentenceTransformer
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
if __name__ == '__main__':
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model, "vector_representations_large_mean_threaded", True)

    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbedding.pretrain_dataset(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("non threaded", time_elapsed)

    invertedIndex = InvertedIndex(documentEmbedding)
    #
    invertedIndex.kMeansCluster(10)

    evaluation1 = evaluation(invertedIndex, False)

    print(evaluation1)

    while True:
        pass
    #
    documentEmbedding1 = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_small_mean_5_threaded", True)
    print("start pretrain_dataset threaded")
    time_start = time.perf_counter()
    documentEmbedding1.pretrain_dataset_parallel(True, 70)
    time_elapsed = (time.perf_counter() - time_start)
    print("threaded", time_elapsed)

    queryProcessing = QueryProcessing(documentEmbedding)
    queryProcessing1 = QueryProcessing(documentEmbedding1)
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
    invertedIndex1 = InvertedIndex(documentEmbedding1)
    #
    invertedIndex.kMeansCluster(5)
    invertedIndex1.kMeansCluster(5)
    #
    # time_start = time.perf_counter()
    # test_queries(invertedIndex, False)
    # time_elapsed = (time.perf_counter() - time_start)
    # print(time_elapsed)
    evaluation1 = evaluation(invertedIndex, True)
    evaluation2 = evaluation(invertedIndex, True)

    print(evaluation1)
    print(evaluation2)

    assert evaluation1["MAP@1"] == evaluation2["MAP@1"] and evaluation1["MAP@3"] == evaluation2["MAP@3"] and evaluation1["MAP@5"] == evaluation2["MAP@5"] and evaluation1["MAP@10"] == evaluation2["MAP@10"]
    #print("threaded", evaluation(queryProcessing1, False))
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
