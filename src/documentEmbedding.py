import numpy as np
import os
from datetime import datetime
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sympy.core.numbers import Infinity


class DocumentEmbedding:
    def __init__(self,dataset_location, model):
        self.dataset_location = dataset_location
        self.model = model
        self.folders = {}
        self.doc_vectors = []
        self.document_embeddings = "../documtent_embeddings"
        if not os.path.exists(self.dataset_location):
            raise Exception("ERROR: path not found:" + self.dataset_location)
        if not os.path.isdir(self.document_embeddings):
            os.makedirs(self.document_embeddings)
        if os.path.isfile(self.document_embeddings+"/vector_representations"):
            self.doc_vectors = pickle.load(open(self.document_embeddings+"/vector_representations","rb"))


    def pretrain_dataset(self, reindex=False):
        if reindex or not os.path.isfile(self.document_embeddings+"/vector_representations"):
            print("indexing path: ", self.dataset_location)
            self.doc_vectors = []
            for d in os.listdir(self.dataset_location):
                if d.endswith('.txt'):
                    file_path = os.path.join(self.dataset_location, d)
                    self.__index_document(file_path)
            pickle.dump(self.doc_vectors, open(self.document_embeddings+"/vector_representations","wb"))


    def __index_document(self, path):
        with open(path, "r") as f:
            doc_text = f.read()
            self.doc_vectors.append((path,np.array(self.model.encode(doc_text))))

