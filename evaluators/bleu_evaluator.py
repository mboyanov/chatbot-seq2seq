from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu as bleu

from evaluators.evaluators import Evaluator


class BLEUEvaluator(Evaluator):
    """"
    Evaluates average BLEU score for the response - answers
    """
    def __init__(self):
        self.total_bleu = 0.0
        self.total_docs = 0.0
        self.smoothing_function = SmoothingFunction().method2

    def update(self, question, response, answers, vectorizer):
        import re
        answers = [a[0] for a in answers if a[1]]
        answers = [re.split("\s", a) for a in answers]
        response = re.split("\s", response)
        try:
            bleu_score = bleu(answers, response, smoothing_function=self.smoothing_function)
        except ZeroDivisionError:
            bleu_score = 0.0
            print("Bleu score 0 for response %s" % str(response))
        self.total_docs += 1
        self.total_bleu += bleu_score

    def results(self):
        return self.total_bleu / self.total_docs