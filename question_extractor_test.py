from question_extractor import QuestionExtractor

import unittest

class TestQuestionExtractor(unittest.TestCase):

    def test_should_return_unchanged_when_length_is_less_than_max(self):
        question_extractor = QuestionExtractor(max_len=50)
        q = "What is the meaning of life?"
        res = question_extractor.extract(q)
        self.assertEqual(q, res)

    def test_should_extract_question(self):
        question_extractor = QuestionExtractor(max_len=7)
        q = "I don't know What is the meaning of life?"
        res = question_extractor.extract(q)
        self.assertEqual("what is the meaning of life ?", res)
