import math
from collections import defaultdict

import numpy as np

class BM25:


    def __init__(self, tokenizer, k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0.0
        self.N = 0
        self.occurrences = defaultdict(int)


    def fit(self, documents):
        self.N += len(documents)
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            self.avg_doc_len += len(tokens)
            for token in set(tokens):
                self.occurrences[token] += 1
        self.avg_doc_len /= self.N


    def transform(self, query, documents):
        res = np.zeros(len(documents))
        query_tokens = set(self.tokenizer.tokenize(query))

        for i, document in enumerate(documents):
            document_tokens = self.tokenizer.tokenize(document)
            for token in document_tokens:
                if token in query_tokens:
                    res[i] += self.idf(token) * (self.k1 + 1) / \
                              (1 + self.k1 * (1 - self.b + self.b * len(document_tokens)/self.avg_doc_len))
        return res


    def idf(self, token):
        return math.log(self.N - self.occurrences[token] + 0.5) - math.log(self.occurrences[token] + 0.5)
