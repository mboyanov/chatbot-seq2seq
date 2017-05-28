from evaluators.evaluators import Evaluator
from data_utils_udc import default_tokenizer
class LengthEvaluator(Evaluator):
    """
    Calculates average type-token ratio
    """
    def __init__(self):
        self.total_docs = 0
        self.length = 0.0

    def update(self, question, response, answers, vectorizer):
        response = default_tokenizer.tokenize(response)
        self.total_docs += 1
        self.length += len(response)

    def results(self):
        return self.length/self.total_docs

