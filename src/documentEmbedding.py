import concurrent.futures
import heapq
from operator import itemgetter

import numpy as np
import os
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from heapq import heappush, nlargest
from src.queryProcessor import QueryProcessor

class DocumentEmbedding(QueryProcessor):
    def __init__(self,dataset_location:str, model:SentenceTransformer, file_name:str, mean_encodings:bool = False, mean_overlap:int = 20, mean_lenght:int = 256):
        self.dataset_location = dataset_location
        self.model = model
        self.folders = {}
        self.doc_vectors = []
        self.save_folder = "../document_embeddings"
        self.file_name = "/" + file_name
        if mean_encodings:
            self.file_name += "_mean_" + str(mean_overlap) + "_" + str(mean_lenght)
        self.mean_encodings = mean_encodings
        self.mean_overlap = mean_overlap
        self.mean_lenght = mean_lenght
        if not os.path.exists(self.dataset_location):
            raise Exception("ERROR: path not found:" + self.dataset_location)
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        if os.path.isfile(self.save_folder + self.file_name):
            self.doc_vectors = pickle.load(open(self.save_folder + self.file_name, "rb"))


    def pretrain_dataset(self, reindex:bool=False):
        counter = 0
        if reindex or not os.path.isfile(self.save_folder + self.file_name):
            print("indexing path: ", self.dataset_location)
            self.doc_vectors = []
            for d in os.listdir(self.dataset_location):
                if d.endswith('.txt'):
                    file_path = os.path.join(self.dataset_location, d)
                    self.__index_document(file_path, self.doc_vectors, self.model.tokenizer)
                if counter%1 == 0:
                    print(counter)
                counter+=1
            pickle.dump(self.doc_vectors, open(self.save_folder + self.file_name, "wb"))

    def __process_chunk(self, chunk, thread_id):
        chunk_vectors = []
        counter = 0
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        for d in chunk:
            if d.endswith('.txt'):
                file_path = os.path.join(self.dataset_location, d)
                self.__index_document(file_path, chunk_vectors, tokenizer)
            if counter % 100 == 0:
                print("thread", thread_id, "count:", counter, "/", len(chunk))
            counter += 1
        return chunk_vectors

    def pretrain_dataset_parallel(self, reindex:bool=False, threads_cnt:int=20):
        if reindex or not os.path.isfile(self.save_folder + self.file_name):
            print("Parallel indexing path: ", self.dataset_location)

            files = [d for d in os.listdir(self.dataset_location) if d.endswith('.txt')]
            chunks_cnt = int(len(files) / threads_cnt)
            chunks = [files[i:i + chunks_cnt] for i in range(0, len(files), chunks_cnt)]

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads_cnt)
            all_thread_results = []
            for chunk in chunks:
                all_thread_results.append(pool.submit(self.__process_chunk, chunk, len(all_thread_results)))

            pool.shutdown(wait=True)

            print(all_thread_results)
            for result in all_thread_results:
                self.doc_vectors.extend(result.result())

            # Save combined vectors
            pickle.dump(self.doc_vectors, open(self.save_folder + self.file_name, "wb"))

    def __get_mean_encoding(self, content, length, overlap, tokenizer):
        chunks = []
        tokens = tokenizer.encode(content, truncation=False)
        for i in range(0, len(tokens), length - overlap):
            chunks.append(tokenizer.decode(tokens[i:i + length]))

        encodings = self.model.encode(chunks)
        return np.mean(encodings, axis=0)

    def __index_document(self, path, array_to_add, tokenizer):
        with open(path, "r") as f:
            doc_text = f.read()

        if self.mean_encodings and doc_text != "":
            # summary_text = self.summarize_large_document(doc_text)
            array_to_add.append(tuple((path, self.__get_mean_encoding(doc_text, self.mean_lenght, self.mean_overlap, tokenizer))))
        elif doc_text != "":
            # summary_text = self.summarize_large_document(doc_text)
            array_to_add.append(tuple((path, np.array(self.model.encode(doc_text)))))


    def processQueryLoop(self, query: str, k:int):
        best_matches = []
        query_vector = np.array(self.model.encode(query))
        for doc_path_vector in self.document_embedding.doc_vectors:
            doc_path = doc_path_vector[0]
            doc_vector = doc_path_vector[1]
            cos_sim = cosine_similarity(query_vector.reshape(1, len(query_vector)), doc_vector.reshape(1, len(query_vector)))
            heappush(best_matches, (cos_sim, doc_path))

        return [match[1] for match in nlargest(k, best_matches, itemgetter(0))]

    def processQuery(self, query: str, k:int):
        query_vector = np.array(self.model.encode(query))
        cos_similarities = cosine_similarity(query_vector.reshape(1, len(query_vector)), [e[1] for e in self.doc_vectors])[0]
        result_indexes = nlargest(k,enumerate(cos_similarities), itemgetter(1))
        return [self.doc_vectors[match[0]][0]for match in result_indexes]