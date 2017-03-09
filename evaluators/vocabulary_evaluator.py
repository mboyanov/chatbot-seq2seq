from evaluators.evaluators import Evaluator
from tokenizer.tokenizer import Tokenizer
import data_utils_udc

class VocabularyEvaluator(Evaluator):

    def __init__(self, tokenizer=Tokenizer(data_utils_udc.UNK_ID)):
        self.vocab = set()
        self.target_vocab = set()
        self.tokenizer = tokenizer

    def update(self, question, response, answers, vectorizer):
        answers = [a[0] for a in answers if a[1]]
        for w in self.tokenizer.tokenize(response):
            self.vocab.add(w)
        for answer in answers:
            for w in self.tokenizer.tokenize(answer):
                self.target_vocab.add(w)

    def results(self):
        return (("Chatbot Vocabulary size", len(self.vocab)),
                ("Target Vocabulary size", len(self.target_vocab)),
                ("Intersection ", len(self.vocab.intersection(self.target_vocab))))