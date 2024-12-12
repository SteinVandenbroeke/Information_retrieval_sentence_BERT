import heapq
from operator import itemgetter

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from heapq import heappush, nlargest


class QueryProcessor:
    def __init__(self):
        self.file_name = None

    def processQuery(self, query, k):
        raise NotImplementedError()