from match_pairs_mongo_importer import flat2dict
import unittest

ex = ['Q1_R1', 'Q1_R1_C1', 'massage oil', 'kahrama national area .', '0.24406854', '0.12151218 0.16187566 0.22241618 -2.0',
      'massage oil', 'it is right behind kahrama in the national area .', 'massage oil is there any place i can find scented massage oils in qatar?', 'yes. it is right behind kahrama in the national area.\n']
dict_format_ex= flat2dict(ex, 'test')

class TestQuestionExtractor(unittest.TestCase):


    def test_should_have_question_id(self):
        self.assertEqual(dict_format_ex['question_id'], "Q1_R1")

    def test_should_have_comment_id(self):
        self.assertEqual(dict_format_ex['comment_id'], "Q1_R1_C1")

    def test_should_have_relevance(self):
        self.assertAlmostEqual(dict_format_ex['relevance'], 0.24406854)

    def test_should_have_question_sentence(self):
        self.assertEqual(dict_format_ex['question_sentence'], "massage oil")
    def test_should_have_comment_sentence(self):
        self.assertEqual(dict_format_ex['comment_sentence'], "it is right behind kahrama in the national area .")

    def test_should_have_dataset(self):
        self.assertEqual(dict_format_ex['dataset'], 'test')