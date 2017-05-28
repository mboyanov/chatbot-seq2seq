from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu as bleu

from evaluators.evaluators import Evaluator
from evaluators.map_helper import calculateMAP
import re
import numpy as np


class BLEUSingleEvaluator(Evaluator):
    """"
    Evaluates average BLEU score for the response - answers
    """
    def __init__(self):
        self.total_docs = 0.0
        self.smoothing_function = SmoothingFunction().method2
        self.mapBLEU = 0.0
        self.total_average_bleu = 0.0

    def update(self, question, response, answers, vectorizer):
        all_answers = [a[0] for a in answers]
        all_answers = [re.split("\s+", a) for a in all_answers]
        response = re.split("\s+", response)
        similarities = []
        for a in all_answers:
            try:
                bleu_score = bleu([a], response, smoothing_function=self.smoothing_function)
            except ZeroDivisionError:
                bleu_score = 0.0
            similarities.append(bleu_score)

        map_score_based_on_bleu = calculateMAP(similarities, [a[1] for a in answers])
        average_bleu = np.mean(similarities)
        self.total_docs += 1
        self.mapBLEU += map_score_based_on_bleu
        self.total_average_bleu += average_bleu
        return similarities

    def results(self):
        return {
            "MAP-BLEU" : self.mapBLEU / self.total_docs,
            "meanAvgBLEU": self.total_average_bleu / self.total_docs
        }