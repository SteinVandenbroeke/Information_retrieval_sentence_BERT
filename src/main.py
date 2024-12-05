from documentEmbedding import DocumentEmbedding
from queryProcessing import QueryProcessing
from testQueries import *
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documentEmbedding = DocumentEmbedding("../../datasets/full_docs_small", model)
documentEmbedding.pretrain_dataset(False)

queryProcessing = QueryProcessing(documentEmbedding)

test_queries(queryProcessing,True)
#print(evaluation(queryProcessing,True))
#print(queryProcessing.processQuery("what agency can i report a scammer concerning my computer", 5))