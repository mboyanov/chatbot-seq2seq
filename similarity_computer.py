from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict


def computeSimilarities(answers, response, vectorizer):
    response_transformed = vectorizer.transform([response])
    answers_transformed = vectorizer.transform(answers)
    distances = cosine_distances(response_transformed, answers_transformed)
    return [1 - x for x in distances[0]]

class SimilarityComputer():

    def __init__(self, additional_vectorizers):
        self.similarities = defaultdict(dict)
        self.additional_vectorizers = additional_vectorizers


    def onThread(self, q, answers, response, responses_to_comments =None):
        q = q['text']
        for additional_vectorizer in self.additional_vectorizers:
            sims = computeSimilarities([a[0] for a in answers], response, additional_vectorizer['vectorizer'])
            self.addSimilarityFeatures(answers, sims, additional_vectorizer['label'])
            if responses_to_comments is not None:
                sims_gen = computeSimilarities(responses_to_comments, q, additional_vectorizer['vectorizer'])
                self.addSimilarityFeatures(answers, sims_gen,
                                      '%s-generated-question' % additional_vectorizer['label'])
        return [self.similarities[a[2]] for a in answers]

    def addSimilarityFeatures(self, answers, sims, label):
        sorted_by_sim = sorted(zip(answers, sims), key=lambda x: x[1], reverse=True)
        arihmetic_rank_quotient = 1 / (len(answers))
        for i, (a, sim) in enumerate(sorted_by_sim):
            self.similarities[a[2]][label] = sim
            self.similarities[a[2]]["%s-reciprocal-rank" % label] = 1 / (i + 1)
            self.similarities[a[2]]["%s-arithmetic-rank" % label] = 1 - i * arihmetic_rank_quotient


class Vectorizer:

    def transform(self, X):
        if len(X) == 1:
            return [[1,0,1]]
        return [[1,0,1], [1,0,0]]

vectorizers = [{
    'vectorizer': Vectorizer(),
    'label': 'test'
}]

computeSimilarities([[1,0,1], [1,0,0]], [1,0,1], Vectorizer())