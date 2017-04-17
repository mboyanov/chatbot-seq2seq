import numpy as np
from gensim.models.word2vec import Word2Vec


class EmbeddingLookup():

    def __init__(self, embedding_path):
        model = Word2Vec.load(embedding_path)
        self.embeddings = model.wv
        del model

    def transform(self, X):
        result = np.zeros((len(X), 100))
        for i, x in enumerate(X):
            try:
                words = [w.lower() for w in x.split() if w in self.embeddings]
                if len(words) > 0:
                    emb = self.embeddings[words]
                    result[i] = np.sum(emb, axis=0)
            except Exception as e:
                print(x, e)
                input()
        return result


