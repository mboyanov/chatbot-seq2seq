from evaluators.evaluators import Evaluator
from tokenizer import tokenizer as default_tokenizer


class VocabularyEvaluator(Evaluator):

    def __init__(self, tokenizer=default_tokenizer):
        self.vocab = set()
        self.target_vocab = set()
        self.tokenizer = tokenizer

    def update(self, question, response, answers, vectorizer):
        answers = [a[0] for a in answers if a[1]]
        for w in self.tokenizer(response):
            self.vocab.add(w)
        for answer in answers:
            for w in self.tokenizer(answer):
                self.target_vocab.add(w)

    def results(self):
        return (("Chatbot Vocabulary size", len(self.vocab)),
                ("Target Vocabulary size", len(self.target_vocab)),
                ("Intersection ", len(self.vocab.intersection(self.target_vocab))))