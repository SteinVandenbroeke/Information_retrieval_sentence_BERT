import numpy as np
import os
from datetime import datetime
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sympy.core.numbers import Infinity


class DocumentEmbedding:
    def __init__(self,dataset_location, model, pre_save_path, mean_encodings = False):
        self.dataset_location = dataset_location
        self.model = model
        self.folders = {}
        self.doc_vectors = []
        self.document_embeddings = "../document_embeddings"
        self.pre_save_path = "/" + pre_save_path
        self.mean_encodings = mean_encodings
        if not os.path.exists(self.dataset_location):
            raise Exception("ERROR: path not found:" + self.dataset_location)
        if not os.path.isdir(self.document_embeddings):
            os.makedirs(self.document_embeddings)
        if os.path.isfile(self.document_embeddings+self.pre_save_path):
            self.doc_vectors = pickle.load(open(self.document_embeddings+self.pre_save_path,"rb"))


    def pretrain_dataset(self, reindex=False):
        counter = 0
        if reindex or not os.path.isfile(self.document_embeddings+self.pre_save_path):
            print("indexing path: ", self.dataset_location)
            self.doc_vectors = []
            for d in os.listdir(self.dataset_location):
                if d.endswith('.txt'):
                    file_path = os.path.join(self.dataset_location, d)
                    self.__index_document(file_path)
                if counter%1000 == 0:
                    print(counter)
                counter+=1
            pickle.dump(self.doc_vectors, open(self.document_embeddings+self.pre_save_path,"wb"))


    def __get_mean_encoding(self, content, length, overlap):
        if len(content) < length:
            np.array(self.model.encode(content))

        encodings = []
        for i in range(0, len(content), length - overlap):
            chunk = content[i:i + length]
            if len(chunk) <= 0:
                assert ValueError("not enough items")
            encoding = np.array(self.model.encode(chunk))
            if np.isnan(encoding).any() or np.isinf(encoding).any():
                print("Array contains NaN or Inf values.")

            encodings.append(encoding)

        if len(encodings) <= 0:
            assert ValueError("not enough items")

        return np.mean(encodings, axis=0)

    def __index_document(self, path):
        with open(path, "r") as f:
            doc_text = f.read()
            if self.mean_encodings:
                self.doc_vectors.append((path, self.__get_mean_encoding(doc_text, 512, 50)))
            else:
                self.doc_vectors.append((path, np.array(self.model.encode(doc_text))))

