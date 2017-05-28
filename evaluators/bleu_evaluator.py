from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu as bleu

from evaluators.evaluators import Evaluator
import re


class BLEUEvaluator(Evaluator):
    """"
    Evaluates average BLEU score for the response - answers
    """
    def __init__(self):
        self.total_bleu = 0.0
        self.total_docs = 0.0
        self.smoothing_function = SmoothingFunction().method2
        self.total_bleu_all = 0.0

    def update(self, question, response, answers, vectorizer):
        correct_answers = [a[0] for a in answers if a[1]]
        correct_answers = [re.split("\s+", a) for a in correct_answers]
        all_answers = [a[0] for a in answers]
        all_answers = [re.split("\s+", a) for a in all_answers]
        response = re.split("\s+", response)
        try:
            bleu_score = bleu(correct_answers, response, smoothing_function=self.smoothing_function)
            bleu_score_all = bleu(all_answers, response, smoothing_function=self.smoothing_function)
        except ZeroDivisionError:
            bleu_score = 0.0
            bleu_score_all = 0.0
            print("Bleu score 0 for response %s" % str(response))
        self.total_docs += 1
        self.total_bleu += bleu_score
        self.total_bleu_all += bleu_score_all

    def results(self):
        return {
            'BLEU': self.total_bleu / self.total_docs,
            'BLEU_ALL': self.total_bleu_all / self.total_docs
        }