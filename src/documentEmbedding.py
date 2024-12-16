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
        """
         Does create vector embeddings call the pretrain_dataset or pretrain_dataset_parallel function
        :param dataset_location: location to all the txt files
        :param model: SentenceTransformer model to use
        :param file_name: name to save and load the embeddings
        :param mean_encodings: use mean encoding so file truncation is not needed
        :param mean_overlap: the overlap between the two sentences
        :param mean_lenght: the length of each chunk (best set to size of model)
        """
        self.dataset_location = dataset_location
        self.model = model
        self.folders = {}
        self.doc_vectors = []
        self.save_folder = "../document_embeddings" + "/"
        self.file_name = file_name
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
        """
        generate the vector representation of each document and stores it in a file, as well as in self.doc_vectors
        :param reindex: if true, reindex the document vectors. otherwise do nothing
        :return:
        """
        counter = 0
        if reindex or not os.path.isfile(self.save_folder + self.file_name):
            print("indexing path: ", self.dataset_location)
            self.doc_vectors = []
            for d in os.listdir(self.dataset_location):
                if d.endswith('.txt'):
                    file_path = os.path.join(self.dataset_location, d)
                    self.__index_document(file_path, self.doc_vectors, self.model.tokenizer)
                if counter%100 == 0:
                    print(counter)
                counter+=1
            pickle.dump(self.doc_vectors, open(self.save_folder + self.file_name, "wb"))

    def __process_chunk(self, chunk, thread_id):
        """
        generated the vector representation of a single chunk of a document
        :param chunk: the chunk to work with
        :param thread_id: the id of the thread to assign this task
        :return: the vector representation of the chunk
        """
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
        """
        generate the vector representation of each document in parallel (threading)
        :param reindex: if true, reindex the document vectors. otherwise do nothing
        :param threads_cnt: the amount of threads to use for parallel processing
        :return:
        """
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
        """
        get the vector representation of a document, given the information to apply chunking
        :param content: the content of the document
        :param length: the max length of a chunk
        :param overlap: the overlap between chunks
        :param tokenizer: the tokenizer used to tokenize the content
        :return: the vector representation of the document
        """
        chunks = []
        tokens = tokenizer.encode(content, truncation=False)
        for i in range(0, len(tokens), length - overlap):
            chunks.append(tokenizer.decode(tokens[i:i + length]))

        encodings = self.model.encode(chunks)
        return np.mean(encodings, axis=0)

    def __index_document(self, path, array_to_add, tokenizer):
        """
        calculate the vector representation of a document
        :param path: the path of the document
        :param array_to_add: the array to append to the vector representation to
        :param tokenizer: the tokenizer used to tokenize the content
        :return:
        """
        with open(path, "r") as f:
            doc_text = f.read()

        if self.mean_encodings and doc_text != "":
            array_to_add.append(tuple((path, self.__get_mean_encoding(doc_text, self.mean_lenght, self.mean_overlap, tokenizer))))
        elif doc_text != "":
            array_to_add.append(tuple((path, np.array(self.model.encode(doc_text)))))


    def processQueryLoop(self, query: str, k:int):
        """
        processes a given query with index of embeddings using a loop
        :param query: the query in string format
        :param k: amount of relevant documents to retrieve
        :return: relevant documents given the query
        """
        best_matches = []
        query_vector = np.array(self.model.encode(query))
        for doc_path_vector in self.document_embedding.doc_vectors:
            doc_path = doc_path_vector[0]
            doc_vector = doc_path_vector[1]
            cos_sim = cosine_similarity(query_vector.reshape(1, len(query_vector)), doc_vector.reshape(1, len(query_vector)))
            heappush(best_matches, (cos_sim, doc_path))

        return [match[1] for match in nlargest(k, best_matches, itemgetter(0))]

    def processQuery(self, query: str, k:int):
        """
        processes a given query with index of embeddings
        :param query: the query in string format
        :param k: amount of relevant documents to retrieve
        :return: relevant documents given the query
        """
        query_vector = np.array(self.model.encode(query))
        cos_similarities = cosine_similarity(query_vector.reshape(1, len(query_vector)), [e[1] for e in self.doc_vectors])[0]
        result_indexes = nlargest(k,enumerate(cos_similarities), itemgetter(1))
        return [self.doc_vectors[match[0]][0]for match in result_indexes]