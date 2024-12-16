import numpy as np
from transformers import AutoTokenizer

from documentEmbedding import DocumentEmbedding
from clustertedDocumentEmbedding import ClustertedDocumentEmbedding
from testQueries import *
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt


def part_one():
    """
    Tests embedding and retrieval with functionality of the first part of the assigment on the small dataset.
    """
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
    """
    Tests clustering with different values of c and k (clustering)
    :param documentEmbedding: the embedding where evaluation is performed
    """
    print("start pretrain_dataset")
    time_start = time.perf_counter()
    documentEmbedding.pretrain_dataset_parallel(False)
    time_elapsed = (time.perf_counter() - time_start)
    print("Embedding creation/loading time:", time_elapsed)

    plot_results = {}
    for t, c in [(1, 1), (1, 5), (1, 10), (5, 50), (5, 100), (10, 100), (10, 500), (50, 500), (10, 700), (50, 700)]:
        print(f"Create clusters | n:{t} | k:{c}")
        time_start = time.perf_counter()
        clusterted_doc_embeddings = ClustertedDocumentEmbedding(documentEmbedding)
        clusterted_doc_embeddings.kMeansCluster(c)
        clusterted_doc_embeddings.set_c_value(t)
        time_elapsed = (time.perf_counter() - time_start)
        print("Cluster creation/loading time:", time_elapsed)

        print("Evaluation")
        result = evaluation(clusterted_doc_embeddings, False)
        plot_results[str((t, c)) + ' query time:' + str(result["QueryTime"])] = result

def part_two():
    """
    Tests embedding and retrieval with functionality of the first part of the assigment on the small dataset.
    """
    print("====Part Two====")
    print("----Large dataset----")
    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model,"vector_representations_large", False)
    evaluate_clusterings(documentEmbedding)
    print("\n")

    print("----Large dataset full files----")
    documentEmbedding = DocumentEmbedding("../../datasets/full_docs", model, "vector_representations_large", True)
    evaluate_clusterings(documentEmbedding)
    print("\n")

def find_optimal_mean_offset():
    """
    find the best overlap value for chucking
    :return:
    """
    best_offset = {
        "MAP@1": (-1,-1),
        "MAR@1": (-1,-1),
    }
    for offset in range(0, 110, 10):
        documentEmbeddingSmall = DocumentEmbedding("../../datasets/full_docs_small", model, "vector_representations_small",True, offset)
        print("start pretrain_dataset")
        time_start = time.perf_counter()
        documentEmbeddingSmall.pretrain_dataset_parallel(False)
        time_elapsed = (time.perf_counter() - time_start)
        print("Embedding creation/loading time: ", time_elapsed)
        print(f"Evaluation for offset {offset}")
        evaluation_results = evaluation(documentEmbeddingSmall, True, "best_offset", f"_offset_{offset}")

        for k, v in best_offset.items():
            if evaluation_results[k] >= v[1]:
                best_offset[k] = (offset, evaluation_results[k])


    print("Best offset: ", best_offset)
    return best_offset["MAP@1"][0]

if __name__ == '__main__':
    model = SentenceTransformer("all-MiniLM-L6-v2")
    part_one()
    part_two()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    total = 0
    counter = 0

    #find_optimal_mean_offset()