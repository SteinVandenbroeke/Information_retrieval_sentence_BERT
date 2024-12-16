import heapq
import os
import pickle
from operator import itemgetter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.documentEmbedding import DocumentEmbedding
from src.queryProcessor import QueryProcessor


class ClustertedDocumentEmbedding(QueryProcessor):
    def __init__(self, documentEmbedding: DocumentEmbedding, cluster_depth:int = 1):
        self.documentEmbedding = documentEmbedding
        self.centroids = []
        self.inverted_index = {}
        self.cluster_depth = cluster_depth
        self.file_name = None
        self.c = None

    def kMeansCluster(self, k, reindex=False):
        """
        indexes the embeddings using k-means clustering
        :param k: the amount of clusters
        :param reindex: if true, the index is computed again and overwritten. if false, get index from file and skip
        :return:
        """
        self.file_name = self.documentEmbedding.file_name + "_cluster_" + str(k)
        file_name = self.documentEmbedding.save_folder + self.documentEmbedding.file_name + "_cluster_" + str(k)
        if os.path.isfile(file_name) and not reindex:
            loaded_file = pickle.load(open(file_name, "rb"))
            self.inverted_index = loaded_file["inverted_index"]
            self.centroids = loaded_file["centroids"]
            return

        kmeans = KMeans(n_clusters=k)
        kmeans.fit([e[1] for e in self.documentEmbedding.doc_vectors])

        cluster_labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        for doc_idx, label in enumerate(cluster_labels):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(doc_idx)

        pickle.dump({"inverted_index": self.inverted_index, "centroids": self.centroids}, open(file_name, "wb"))


    def set_c_value(self, c):
        """
        set self.c value
        :param c: c value
        :return:
        """
        self.c = c

    def processQuery(self, query:str, k:int):
        """
        processes a given query with index of embeddings
        :param query: the query in string format
        :param k: amount of relevant documents to retrieve
        :return: relevant documents given the query
        """
        assert self.c is not None
        query_vector = self.documentEmbedding.model.encode([query])
        return self.getDocuments(query_vector, self.c, k)

    def getDocuments(self, query_embedding, c, k, embeddings = None, id_mapping = None, __cluster_depth = None):
        """
        calculates the similarity between query and documents to retrieve the most relevant documents
        :param query_embedding: the vector representation of the query
        :param c: how many clusters to search
        :param k: amount of return doc
        :param embeddings: embeddings to search in
        :param id_mapping:
        :param __cluster_depth: the depth of the index, used for recursive calls
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

        return self.getDocuments(query_embedding, c, k, [self.documentEmbedding.doc_vectors[item][1] for item in relevant_docs], [self.documentEmbedding.doc_vectors[item][0] for item in relevant_docs], __cluster_depth)