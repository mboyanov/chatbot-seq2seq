from sklearn.metrics.pairwise import cosine_distances

from evaluators.evaluators import Evaluator

class MAPEvaluator(Evaluator):
    """
    Calculates MAP
    """
    def __init__(self):
        self.total_docs = 0
        self.meanAvgPrecision = 0.0

    def update(self, question, response, answers, vectorizer):
        answers_transformed = vectorizer.transform([a[0] for a in answers])
        correct = [a[1] for a in answers]
        response = vectorizer.transform([response])
        score = self.calculateMAP(answers_transformed, response, correct)
        self.meanAvgPrecision += score
        self.total_docs += 1

    def calculateMAP(self, answers, target, correct):
        distances = cosine_distances(target, answers)
        results = list(zip(distances[0], correct))
        results.sort(key= lambda x: x[0])
        relevant_docs = 0
        score = 0.0
        for i, r in enumerate(results):
            if r[1]:
                relevant_docs += 1
                score += relevant_docs / (i + 1)
        return score / relevant_docs

    def results(self):
        return self.meanAvgPrecision/self.total_docs