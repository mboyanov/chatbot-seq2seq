
from evaluators.evaluators import Evaluator
from data_utils_udc import default_tokenizer
class TTREvaluator(Evaluator):
    """
    Calculates average type-token ratio
    """
    def __init__(self):
        self.total_docs = 0
        self.ttr = 0.0

    def update(self, question, response, answers, vectorizer):
        response = default_tokenizer.tokenize(response)
        self.total_docs += 1
        if len(response) > 0:
            self.ttr += float(len(set(response))) / len(response)

    def results(self):
        return self.ttr/self.total_docs