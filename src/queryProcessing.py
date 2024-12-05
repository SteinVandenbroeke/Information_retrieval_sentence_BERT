from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from heapq import heappush, nlargest


class QueryProcessing:
    def __init__(self, document_embedding):
        self.model = document_embedding.model
        self.document_embedding = document_embedding


    def processQuery(self, query, k):
        best_matches = []
        query_vector = np.array(self.model.encode(query))
        for doc_path_vector in self.document_embedding.doc_vectors:
            doc_path = doc_path_vector[0]
            doc_vector = doc_path_vector[1]
            cos_sim = cosine_similarity(query_vector.reshape(1, len(query_vector)), doc_vector.reshape(1, len(query_vector)))
            heappush(best_matches, (-cos_sim, doc_path))
            best_matches = best_matches[:k]

        return [match[1] for match in best_matches]