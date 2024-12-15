from documentEmbedding import DocumentEmbedding
from clustertedDocumentEmbedding import ClustertedDocumentEmbedding
from testQueries import *
from sentence_transformers import SentenceTransformer
import time

def part_one():
    print("====Part One====")
    print("----Small dataset----")
    documentEmbeddingSmall = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_small",
                                               False)
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbeddingSmall.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time: ", time_elapsed)
    print("Evaluation")
    evaluation(documentEmbeddingSmall, True)
    print("\n")

    print("----Small dataset full files----")
    documentEmbeddingSmall = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_small", True)
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbeddingSmall.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time: ", time_elapsed)
    print("Evaluation")
    evaluation(documentEmbeddingSmall, True)
    print("\n")

def evaluate_clusterings(documentEmbedding):
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbedding.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time:", time_elapsed)

    for c, k in [(1, 1), (1, 5), (1, 10), (5, 50), (5, 100), (10, 100), (10, 500), (50, 500), (10, 700), (50, 700)]:
        print(f"Create clusters | c:{c} | k:{k}")
        time_start = time.perf_counter()
        clusterted_doc_embeddings = ClustertedDocumentEmbedding(documentEmbedding)
        clusterted_doc_embeddings.kMeansCluster(k)
        clusterted_doc_embeddings.set_c_value(c)
        time_elapsed = (time.perf_counter() - time_start)
        print("Cluster creation/loading time:", time_elapsed)

        print("Evaluation")
        evaluation(clusterted_doc_embeddings, False)

def part_two():
    print("====Part Two====")
    print("----Small dataset----")
    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model,"vector_representations_large", False)
    evaluate_clusterings(documentEmbedding)
    print("\n")

    print("----Large dataset full files----")
    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model, "vector_representations_large", True)
    evaluate_clusterings(documentEmbedding)
    print("\n")

if __name__ == '__main__':
    model = SentenceTransformer("all-MiniLM-L6-v2")
    part_one()
    part_two()