import os

from sklearn.metrics.pairwise import cosine_distances

import QLXMLReaderPy as reader
import data_utils_udc

from collections import defaultdict

ql_home = "/home/martin/data/qatarliving"

kelp_files_home = "/home/martin/projects/tutorial/"
configs = [
    {
        'xml': os.path.join(ql_home, 'train/SemEval2016-Task3-CQA-QL-train-concat-with-multiline.xml'),
        'inOutPair': (
            os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-train-part1-with-multiline.xml.taskA.klp'),
            os.path.join(kelp_files_home,
                         'v10-SemEval2016-Task3-CQA-QL-train-part1-with-multiline-with-bot.xml.taskA.klp')),
        'responsesFile': '/home/martin/projects/machine-translation-scoring/train'
    },
    {
        'xml': os.path.join(ql_home, 'dev/SemEval2016-Task3-CQA-QL-dev-subtaskA-with-multiline.xml'),
        'inOutPair': (os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-dev-with-multiline.xml.taskA.klp'),
                      os.path.join(kelp_files_home,
                                   'v10-SemEval2016-Task3-CQA-QL-dev-with-multiline-with-bot.xml.taskA.klp')),
        'responsesFile': '/home/martin/projects/machine-translation-scoring/v3-dev'
    },
    {
        'xml': os.path.join(ql_home, 'test/SemEval2016-Task3-CQA-QL-test-subtaskA-with-multiline.xml'),
        'inOutPair': (os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-test-with-multiline.xml.taskA.klp'),
                      os.path.join(kelp_files_home,
                                   'v10-SemEval2016-Task3-CQA-QL-test-with-multiline-with-bot.xml.taskA.klp')),
        'responsesFile': '/home/martin/projects/machine-translation-scoring/test'
    },
# {
#         'xml': os.path.join(ql_home, 'dev/SemEval2016-Task3-CQA-QL-passthrough-subtaskA-with-multiline.xml'),
#         'inOutPair': (os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-passthrough-with-multiline.xml.taskA.klp'),
#                       os.path.join(kelp_files_home,
#                                    '/tmp/passthrough')),
#         'responsesFile': '/tmp/passthrough-responses'
#     },
]

data_dir = os.path.join(ql_home, 'matchedPairs_ver5')
vocab_path = os.path.join(data_dir, 'vocab40000.qa')

vocab = data_utils_udc.initialize_tokenizer(vocab_path, False).vocab
default_vectorizer = data_utils_udc.tfidfVectorizer(data_dir, vocab, 'ql')
print("Vectorizer initialized")
bm25 = data_utils_udc.bm25(data_dir, 'ql')
print("bm25 initialized")




from match_pairs_mongo_question_comment_extractor import get_qc_pair
def generateHypothesisReferences(responder, responder_multiple, dp, questionExtractor= lambda x: x):
    """

    :param responder:
    :param dp:
    :return:
    """
    responses = {}
    for q, answers in reader.read(dp):
        questions_text = [q['text']]*10
        answers_text = [a[0] for a in answers]
        answer_ids = [a[2] for a in answers]
        matched_pairs = [get_qc_pair(x) for x in answer_ids]
        for i in range(len(matched_pairs)):
            if matched_pairs[i] is not None:
                answers_text[i] = matched_pairs[i]['comment_sentence']
                questions_text[i] = matched_pairs[i]['question_sentence']
        generated_responses = responder_multiple(questions_text + answers_text)
        for a, rc, rq in zip(answers, generated_responses[len(answer_ids):], generated_responses[0:len(answer_ids)]):
            responses[a[2]] = rq, a[0], rc
    return responses


def persistResponses(responses, fn):
    order = []
    with open('%s-hyp' % fn, 'w') as hyp_out, open('%s-ref' % fn, 'w') as ref_out, open('%s-info' % fn,
                                                                                        'w') as info_out:
        for id, (hyp, ref, _) in responses.items():
            hyp_out.write(hyp + "\n")
            ref_out.write(ref + "\n")
            info_out.write("\t".join([id, hyp, ref]) + "\n")
            order.append(id)
    return '%s-hyp' % fn, '%s-ref' % fn, order

from embedding_lookup import EmbeddingLookup
#from seq2seq_embedding_lookup import Seq2SeqEmbeddingLookup
embedding_vectorizer = EmbeddingLookup("/home/martin/data/qatarliving/embeddings/qatarliving_qc_size100_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin")


from similarity_computer import SimilarityComputer

def getSimilarities(responses, dp, additional_vectorizers):
    similarity_computer = SimilarityComputer(additional_vectorizers)
    for q, answers in reader.read(dp):
        response = responses[answers[0][2]][0]
        responses_to_comments = [responses[a[2]][2] for a in answers]
        similarity_computer.onThread(q,answers, response, responses_to_comments, )
        q = q['text']
        sims2 = bm25.transform(response, [a[0] for a in answers])
        similarity_computer.addSimilarityFeatures(answers, sims2, 'bm25')
        bm25_generated_question_sims = bm25.transform(q, responses_to_comments)
        similarity_computer.addSimilarityFeatures(answers, bm25_generated_question_sims, 'bm25-generated-question')
    return similarity_computer.similarities


def addSimilarityFeatures(answers, sims, similarities, label):
    sorted_by_sim = sorted(zip(answers, sims), key=lambda x: x[1], reverse=True)
    arihmetic_rank_quotient = 1/(len(answers))
    for i, (a, sim) in enumerate(sorted_by_sim):
        similarities[a[2]][label] = sim,
        similarities[a[2]]["%s-reciprocal-rank" % label] = 1/ (i+1)
        similarities[a[2]]["%s-arithmetic-rank" % label] = 1 - i * arihmetic_rank_quotient


sentence = "Good |<||BT:tree|(ROOT (S (NP (NN (massage::n))(NN (oil::n)))(VP (VBZ (be::v)))(ADVP (RB (there::r)))(NP (DT (any::d))(NN (place::n)))(NP (PRP (i::p)))(VP (MD (can::m))(VB (find::v)))(NP (VBN (scent::v))(NN (massage::n))(NNS (oils::n)))(PP (IN (in::i)))(NP (NN (qatar::n)))))|ET||BS:text|massage oil is there any place i can find scented massage oils in qatar?|ES||,||BT:tree|(ROOT (S (VP (VB (try::v)))(NP (DT (both::d)))(NP (-RRB- (}::-))(NNP (i'am::n)))(VP (RB (just::r))(VBG (try::v))(TO (to::t))(VB (be::v)))(ADJP (JJ (helpful::j))))(S (PP (IN (on::i)))(NP (DT (a::d))(JJ (serious::j))(NN (note::n)))(VP (VB (please::v))(VB (go::v)))(ADVP (RB (there::r))))(S (NP (PRP (you::p)))(VP (MD ('ll::m))(VB (find::v)))(NP (WP (what::w)))(NP (PRP (you::p)))(VP (VBP (be::v))(VBG (look::v)))(PP (IN (for::i)))))|ET||BS:text|Try Both ;) I'am just trying to be helpful. On a serious note - Please go there. you'll find what you are looking for.|ES||>| |BDV:WSsim|0.0 0.8005682229995728 0.08966762572526932 0.7889324426651001 0.5595598816871643 0.6260497570037842 |EDV||BDV:features|0.303022 0.172976 0.123091 0.000000 0.000000 0.000000 0.285124 0.237289 0.053376 0.267606 0.387435 0.521127 0.073298 0.085714 0.000000 0.000000 0.000000 0.214286 0.000000 0.151511 0.579746 |EDV||BV:threadFeats|MULT_MID:1.000000 POSITION:0.200000 DIAL_Uq_IN:1.000000 CATEGORY_Qatar_Living_Lounge:1.000000 MULT_REAL:0.200000 LENGTH:0.295000 MULT_BOOL:1.000000 |EV||BS:info|Q1_R1_C5|ES|"
sentence_plus = "Good |<||BT:tree|(ROOT (S (NP (NN (massage::n))(NN (oil::n)))(VP (VBZ (be::v)))(ADVP (RB (there::r)))(NP (DT (any::d))(NN (place::n)))(NP (PRP (i::p)))(VP (MD (can::m))(VB (find::v)))(NP (VBN (scent::v))(NN (massage::n))(NNS (oils::n)))(PP (IN (in::i)))(NP (NN (qatar::n)))))|ET||BS:text|massage oil is there any place i can find scented massage oils in qatar?|ES||,||BT:tree|(ROOT (S (VP (VB (try::v)))(NP (DT (both::d)))(NP (-RRB- (}::-))(NNP (i'am::n)))(VP (RB (just::r))(VBG (try::v))(TO (to::t))(VB (be::v)))(ADJP (JJ (helpful::j))))(S (PP (IN (on::i)))(NP (DT (a::d))(JJ (serious::j))(NN (note::n)))(VP (VB (please::v))(VB (go::v)))(ADVP (RB (there::r))))(S (NP (PRP (you::p)))(VP (MD ('ll::m))(VB (find::v)))(NP (WP (what::w)))(NP (PRP (you::p)))(VP (VBP (be::v))(VBG (look::v)))(PP (IN (for::i)))))|ET||BS:text|Try Both ;) I'am just trying to be helpful. On a serious note - Please go there. you'll find what you are looking for.|ES||>| |BDV:WSsim|0.0 0.8005682229995728 0.08966762572526932 0.7889324426651001 0.5595598816871643 0.6260497570037842 |EDV||BDV:features|0.303022 0.172976 0.123091 0.000000 0.000000 0.000000 0.285124 0.237289 0.053376 0.267606 0.387435 0.521127 0.073298 0.085714 0.000000 0.000000 0.000000 0.214286 0.000000 0.151511 0.579746 1.000000 |EDV||BV:threadFeats|MULT_MID:1.000000 POSITION:0.200000 DIAL_Uq_IN:1.000000 CATEGORY_Qatar_Living_Lounge:1.000000 MULT_REAL:0.200000 LENGTH:0.295000 MULT_BOOL:1.000000 |EV||BS:info|Q1_R1_C5|ES|"
import re

id_regex = r'BS:info\|([^\|]+)\|'
text_regex = r'BS:text\|([^\|]+)\|'
feature_regex = r'BDV:features\|([^\|]+)\|'


def find_id(s):
    matches = re.findall(id_regex, s)
    return matches[0]


def add_features(new_features, s):
    features = re.findall(feature_regex, s)[0]
    for feature in new_features:
        try:
            features += "%.6f " % feature
        except:
            print(feature, new_features, s)
    return re.sub(feature_regex, 'BDV:features|' + features + "|", s)


def find_text(s):
    return re.findall(text_regex, s)


assert (find_id(sentence) == "Q1_R1_C5")
assert (add_features([1.0], sentence) == sentence_plus)

from subprocess import check_output




def getMachineTranslationFeatures(hyp, ref, order):
    output = check_output(["/home/martin/projects/machine-translation-scoring/score.rb", '--hyp', hyp, '--ref', ref, '--individual'])
    print(output)
    results = {}
    with open("%s.individual" % hyp) as result_file:
        for idx, line in enumerate(result_file):
            results[order[idx]] = [float(x) for x in re.split("\s+", line.strip())]
    return results

import tensorflow as tf
from chatbot import decode



active_features = ["tfidf-cosine",
                   "tfidf-cosine-reciprocal-rank",
                   "tfidf-cosine-arithmetic-rank",
                   "bm25",
                   "bm25-reciprocal-rank",
                   "bm25-arithmetic-rank",
                   'embeddings',
                   'embeddings-reciprocal-rank',
                   'embeddings-arithmetic-rank',
                   "tfidf-cosine-generated-question",
                   "tfidf-cosine-generated-question-reciprocal-rank",
                   "tfidf-cosine-generated-question-arithmetic-rank",
                   "bm25-generated-question",
                   "bm25-generated-question-reciprocal-rank",
                   "bm25-generated-question-arithmetic-rank",
                   "embeddings-generated-question",
                   "embeddings-generated-question-reciprocal-rank",
                   "embeddings-generated-question-arithmetic-rank"]

feature_combinations = [
    ["tfidf-cosine",
     "tfidf-cosine-arithmetic-rank",
     "tfidf-cosine-reciprocal-rank"],
    ["bm25",
     "bm25-arithmetic-rank",
     "bm25-reciprocal-rank"],
    ['embeddings',
     "embeddings-arithmetic-rank",
     'embeddings-reciprocal-rank'],
    ["tfidf-cosine-generated-question",
     "tfidf-cosine-generated-question-arithmetic-rank",
     "tfidf-cosine-generated-question-reciprocal-rank"],
    ["bm25-generated-question",
     "bm25-generated-question-arithmetic-rank",
     "bm25-generated-question-reciprocal-rank"],
    ["embeddings-generated-question",
     "embeddings-generated-question-arithmetic-rank",
     "embeddings-generated-question-reciprocal-rank"]
]


with tf.Session() as sess:
    responder, responder_multiple = decode(sess)
    #seq2seq_embedding_vectorizer = Seq2SeqEmbeddingLookup(sess)
    for config in configs:
        print("Processing %s" % config['xml'])
        responses = generateHypothesisReferences(responder, responder_multiple, config['xml'])
        print("Hypotheses generated %s" % config['xml'])
        hyp_file, ref_file, order = persistResponses(responses, config['responsesFile'])
        print("Hypotheses persisted %s" % config['xml'])
        additional_vectorizers = [
            {'vectorizer': default_vectorizer,
             'label': 'tfidf-cosine'
             },
            # { 'vectorizer': seq2seq_embedding_vectorizer,
            #   'label': 'seq2seq_embeddings'
            # },
            {
                'vectorizer': embedding_vectorizer,
                'label': 'embeddings'
            }
        ]
        similarity_features = getSimilarities(responses, config['xml'], additional_vectorizers)
   #     mt_features = getMachineTranslationFeatures(hyp_file, ref_file, order)
        print("Similarities computed %s" % config['xml'])
        inOutPair = config['inOutPair']
        with open(inOutPair[0]) as kelp_in, open(inOutPair[1], 'w') as kelp_out:
            for line in kelp_in:
                c_id = find_id(line)
                if c_id not in similarity_features:
                    print('Sim is none for %s ' % c_id)
                    continue
                feature_dict = similarity_features[c_id]
                feature_list = [feature_dict[x] for x in active_features]
        #        feature_list.extend(mt_features[c_id])
                for feature_combination in feature_combinations:
                    line_plus_feature_combination = add_features([feature_dict[x] for x in feature_combination], line)
                    with open(inOutPair[1]+"."+feature_combination[0], 'a') as kelp_out_feature_combination:
                        kelp_out_feature_combination.write(line_plus_feature_combination)
                line_plus = add_features(feature_list, line)
                kelp_out.write(line_plus)
        print("Kelp files written %s" % config['xml'])
