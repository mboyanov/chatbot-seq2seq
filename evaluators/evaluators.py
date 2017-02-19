class Evaluator():
    """
    Abstract class for evaluators
    """
    def update(self, question, response, answers):
        return NotImplemented()

    def results(self):
        return NotImplemented()


