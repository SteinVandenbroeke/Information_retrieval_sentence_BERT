import heapq
from operator import itemgetter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class InvertedIndex:
    def __init__(self, embeddings, cluster_depth = 1):
        self.embeddings = embeddings.doc_vectors
        self.centroids = []
        self.inverted_index = {}
        self.cluster_depth = cluster_depth
        self.model = embeddings.model

    def kMeansCluster(self, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit([e[1] for e in self.embeddings])

        cluster_labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        for doc_idx, label in enumerate(cluster_labels):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(doc_idx)

    def computeCentroid(self, cluster):
        pass

    def processQuery(self, query, k):
        query_vector = self.model.encode([query])
        return self.getDocuments(query_vector, 5, k)

    def getDocuments(self, query_embedding, c, k, embeddings = None, id_mapping = None, __cluster_depth = None):
        """

        :param query_embedding:
        :param c: how many clusters to search
        :param k: amount of return doc
        :param embeddings: embeddings to search in
        :param __cluster_depth:
        :return:
        """
        if embeddings is None:
            embeddings = self.centroids
        if __cluster_depth is None:
            __cluster_depth = self.cluster_depth
        if id_mapping is None:
            id_mapping = self.inverted_index

        similarities = cosine_similarity(query_embedding, embeddings)[0]
        if __cluster_depth <= 0:
            similarities_indexes = heapq.nlargest(k, enumerate(similarities), itemgetter(1))
            return [id_mapping[match[0]] for match in similarities_indexes]

        similarities_indexes = heapq.nlargest(c, enumerate(similarities), itemgetter(1))
        relevant_docs = []
        #for cluster_id in best_clusters:
            # Retrieve relevant documents from the closest cluster
        for cluster_id_sim in similarities_indexes:
            relevant_docs += id_mapping[cluster_id_sim[0]]

        __cluster_depth -= 1

        return self.getDocuments(query_embedding, c, k, [self.embeddings[item][1] for item in relevant_docs], [self.embeddings[item][0] for item in relevant_docs], __cluster_depth)