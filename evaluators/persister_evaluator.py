from evaluators.evaluators import Evaluator


class PersisterEvaluator(Evaluator):

    def __init__(self, path):
        self.path = path
        self.inputs = []

    def update(self, question, response, answers, vectorizer):
        answers = [a[0] for a in answers if a[1]]
        self.inputs.append((question['text'], response, answers))

    def results(self):
        with open(self.path, 'w') as out:
            for (q, r, answers) in self.inputs:
                out.write("\t".join(["Utterance:", q]) + "\n")
                out.write("\t".join(["Response:", r]) + "\n")
                out.write("\t".join(["Expected:", "::".join(answers)]) + "\n")