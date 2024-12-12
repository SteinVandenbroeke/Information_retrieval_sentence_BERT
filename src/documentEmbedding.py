import concurrent.futures

import numpy as np
import os
import pickle

from transformers import AutoTokenizer


class DocumentEmbedding:
    def __init__(self,dataset_location, model, pre_save_path, mean_encodings = False, mean_overlap = 20, mean_lenght = 256):
        self.dataset_location = dataset_location
        self.model = model
        self.folders = {}
        self.doc_vectors = []
        self.document_embeddings = "../document_embeddings"
        self.pre_save_path = "/" + pre_save_path
        self.mean_encodings = mean_encodings
        self.mean_overlap = mean_overlap
        self.mean_lenght = mean_lenght
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
                    self.__index_document(file_path, self.doc_vectors, self.model.tokenizer)
                if counter%1 == 0:
                    print(counter)
                counter+=1
            pickle.dump(self.doc_vectors, open(self.document_embeddings+self.pre_save_path,"wb"))

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

    def pretrain_dataset_parallel(self, reindex=False, chunks_cnt=30000):
        if reindex or not os.path.isfile(self.document_embeddings + self.pre_save_path):
            print("Parallel indexing path: ", self.dataset_location)

            files = [d for d in os.listdir(self.dataset_location) if d.endswith('.txt')]
            chunks = [files[i:i + chunks_cnt] for i in range(0, len(files), chunks_cnt)]

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)
            all_thread_results = []
            for chunk in chunks:
                all_thread_results.append(pool.submit(self.__process_chunk, chunk, len(all_thread_results)))

            pool.shutdown(wait=True)

            print(all_thread_results)
            for result in all_thread_results:
                self.doc_vectors.extend(result.result())

            # Save combined vectors
            pickle.dump(self.doc_vectors, open(self.document_embeddings + self.pre_save_path, "wb"))

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
            array_to_add.append(tuple((path, self.__get_mean_encoding(doc_text, self.mean_lenght, self.mean_overlap, tokenizer))))
        elif doc_text != "":
            array_to_add.append(tuple((path, np.array(self.model.encode(doc_text)))))

