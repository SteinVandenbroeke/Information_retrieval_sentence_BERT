import numpy as np
import os
from datetime import datetime
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sympy.core.numbers import Infinity


class DocumentEmbedding:
    def __init__(self,dataset_location):
        self.indexing_path = dataset_location
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.folders = {}
        self.doc_vectors = []
        self.document_embeddings = "../documtent_embeddings"
        if not os.path.exists(self.indexing_path):
            raise Exception("ERROR: path not found:" + self.indexing_path)
        if not os.path.isdir(self.document_embeddings):
            os.makedirs(self.document_embeddings)
        if os.path.isfile(self.document_embeddings+"/vector_representations"):
            self.doc_vectors = pickle.load(open(self.document_embeddings+"/vector_representations","rb"))


    def pretrain_dataset(self, reindex=False):
        if reindex or not os.path.isfile(self.document_embeddings+"/vector_representations"):
            print("indexing path: ", self.indexing_path)
            self.doc_vectors = []
            for d in os.listdir(self.indexing_path):
                if d.endswith('.txt'):
                    file_path = os.path.join(self.indexing_path, d)
                    self.__index_document(file_path)
            pickle.dump(self.doc_vectors, open(self.document_embeddings+"/vector_representations","wb"))

        query_vector = np.array(self.model.encode("types of road hugger tires"))
        best_match = 9999999
        best_vector = None
        print(len(self.doc_vectors))
        for i in self.doc_vectors:
            pathh = i[0]
            x = i[1]
            coss = abs(cosine_similarity(query_vector.reshape(1, len(query_vector)), x.reshape(1, len(query_vector))))
            if coss < best_match:
                best_match = coss
                best_vector = pathh
        print(best_match, best_vector)


    def __index_document(self, path):
        with open(path, "r") as f:
            doc_text = f.read()
            self.doc_vectors.append((path,np.array(self.model.encode(doc_text))))

