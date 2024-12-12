from documentEmbedding import DocumentEmbedding
from clustertedDocumentEmbedding import ClustertedDocumentEmbedding
from testQueries import *
from sentence_transformers import SentenceTransformer
import time

def part_one():
    print("====Part One====")
    documentEmbeddingSmall = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_small", True)
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbeddingSmall.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time: ", time_elapsed)
    print("Evaluation")
    evaluation(documentEmbeddingSmall, True)
    print("\n")

def part_two():
    print("====Part Two====")
    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model,"vector_representations_large", True)
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbedding.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time:", time_elapsed)

    for c, k in [(5,100),(5,500),(10,100),(50,500),(50,500)]:
        print("Create clusters")
        time_start = time.perf_counter()
        clusterted_doc_embeddings = ClustertedDocumentEmbedding(documentEmbedding)
        clusterted_doc_embeddings.set_c_value(c)
        clusterted_doc_embeddings.kMeansCluster(k)
        time_elapsed = (time.perf_counter() - time_start)
        print("Cluster creation/loading time:", time_elapsed)

        print("Evaluation")
        evaluation(clusterted_doc_embeddings, False)


    print("\n")


if __name__ == '__main__':
    model = SentenceTransformer("all-MiniLM-L6-v2")
    part_one()
    part_two()