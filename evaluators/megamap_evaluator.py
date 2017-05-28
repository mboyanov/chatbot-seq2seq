
from evaluators.evaluators import Evaluator
from similarity_computer import SimilarityComputer
from collections import defaultdict
from evaluators.map_helper import calculateMAP
from evaluators.bleu_single_evaluator import BLEUSingleEvaluator
import data_utils_udc
import os
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

ql_home = "/home/martin/data/qatarliving"
data_dir = os.path.join(ql_home, 'matchedPairs_ver5')
bm25 = data_utils_udc.bm25(data_dir, 'ql')

class MegaMAPEvaluator(Evaluator):
    """
    Calculates MAP
    """
    def __init__(self, additional_vectorizers):
        self.total_docs = 0
        self.meanAvgPrecision = defaultdict(lambda:0)
        self.similarity_computer = SimilarityComputer(additional_vectorizers)
        self.labels = [vec['label'] for vec in additional_vectorizers] + ["bm25"] + ["tfidf-question"]
        self.bleu_single_evaluator = BLEUSingleEvaluator()

    def update(self, question, response, answers, vectorizer):
        self.similarity_computer.onThread(question, answers, response)
        sims2 = bm25.transform(response, [a[0] for a in answers])
        self.similarity_computer.addSimilarityFeatures(answers, sims2, 'bm25')
        question = vectorizer.transform([question['text']])
        distances_q = cosine_distances(question, vectorizer.transform([a[0] for a in answers]))[0]
        sims_q = [1-x for x in distances_q]
        self.similarity_computer.addSimilarityFeatures(answers, sims_q, 'tfidf-question')
        sims_bleu = self.bleu_single_evaluator.update(question, response, answers, vectorizer)
        self.similarity_computer.addSimilarityFeatures(answers, sims_bleu, 'bleu')
        similarities = [self.similarity_computer.similarities[a[2]] for a in answers]
        correct = [a[1] for a in answers]
        for label in self.labels:
            sims = [sim[label] for sim in similarities]
            score = calculateMAP(sims, correct)
            self.meanAvgPrecision[label] += score
            if label != 'tfidf-question':
                sims_summed = [sim[label] + sim['tfidf-question'] for sim in similarities]
                score = calculateMAP(sims_summed, correct)
                self.meanAvgPrecision[label+"_SUMMED"] += score
        sims_bleu_mapped_sum = [sim['bleu'] + sim['tfidf-question'] + sim['tfidf-cosine'] for sim in similarities]
        score = calculateMAP(sims_bleu_mapped_sum, correct)
        self.meanAvgPrecision["bleu_map_SUMMED"] += score
        sims_map_avg = [np.mean([sim['tfidf-cosine'], sim['bm25'], sim['embeddings'], sim['tfidf-question']]) for sim in similarities]
        score = calculateMAP(sims_map_avg, correct)
        self.meanAvgPrecision["MAP_AVG"] += score
        self.total_docs += 1



    def results(self):
        return {key: self.meanAvgPrecision[key]/self.total_docs for key in self.meanAvgPrecision}