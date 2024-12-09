from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class InvertedIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings.doc_vectors
        self.centroids = []
        self.inverted_index = {}

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