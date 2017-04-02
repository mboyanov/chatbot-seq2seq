from sklearn.metrics.pairwise import cosine_distances

from evaluators.evaluators import Evaluator

class MAPEvaluatorSummed(Evaluator):
    """
    Calculates MAP with respect to the sum of the distance to the response and the question
    """
    def __init__(self):
        self.total_docs = 0
        self.meanAvgPrecision = 0.0
        self.meanAvgPrecisionQ = 0.0

    def update(self, question, response, answers, vectorizer):
        answers_transformed = vectorizer.transform([a[0] for a in answers])
        correct = [a[1] for a in answers]
        response = vectorizer.transform([response])
        question = vectorizer.transform([question])
        score = self.calculateMAP(answers_transformed, response, question, correct)
        self.meanAvgPrecision += score
        self.total_docs += 1

    def calculateMAP(self, answers, response, question, correct):
        distances = cosine_distances(response, answers)
        distances_q = cosine_distances(question, answers)
        distances_sum = distances + distances_q
        results = list(zip(distances_sum[0], correct))
        results.sort()
        relevant_docs = 0
        score = 0.0
        for i, r in enumerate(results):
            if r[1]:
                relevant_docs += 1
                score += relevant_docs / (i + 1)
        return score / relevant_docs

    def results(self):
        return self.meanAvgPrecision/self.total_docs