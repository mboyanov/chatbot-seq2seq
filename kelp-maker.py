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
                         'v4-SemEval2016-Task3-CQA-QL-train-part1-with-multiline-with-bot.xml.taskA.klp')),
        'responsesFile': '/home/martin/projects/machine-translation-scoring/train'
    },
    # {
    #     'xml': os.path.join(ql_home, 'dev/SemEval2016-Task3-CQA-QL-dev-subtaskA-with-multiline.xml'),
    #     'inOutPair': (os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-dev-with-multiline.xml.taskA.klp'),
    #                   os.path.join(kelp_files_home,
    #                                'v4-SemEval2016-Task3-CQA-QL-dev-with-multiline-with-bot.xml.taskA.klp')),
    #     'responsesFile': '/home/martin/projects/machine-translation-scoring/v3-dev'
    # },
    # {
    #     'xml': os.path.join(ql_home, 'test/SemEval2016-Task3-CQA-QL-test-subtaskA-with-multiline.xml'),
    #     'inOutPair': (os.path.join(kelp_files_home, 'SemEval2016-Task3-CQA-QL-test-with-multiline.xml.taskA.klp'),
    #                   os.path.join(kelp_files_home,
    #                                'v4-SemEval2016-Task3-CQA-QL-test-with-multiline-with-bot.xml.taskA.klp')),
    #     'responsesFile': '/home/martin/projects/machine-translation-scoring/test'
    # }
]

data_dir = os.path.join(ql_home, 'matchedPairs_ver5')
vocab_path = os.path.join(data_dir, 'vocab40000.qa')

vocab = data_utils_udc.initialize_tokenizer(vocab_path, False).vocab
default_vectorizer = data_utils_udc.tfidfVectorizer(data_dir, vocab, 'ql')
print("Vectorizer initialized")
bm25 = data_utils_udc.bm25(data_dir, 'ql')
print("bm25 initialized")


def computeSimilarities(answers, response, vectorizer=default_vectorizer):
    response_transformed = vectorizer.transform([response])
    answers_transformed = vectorizer.transform(answers)
    distances = cosine_distances(response_transformed, answers_transformed)
    return [1 - x for x in distances[0]]


def generateHypothesisReferences(responder, responder_multiple, dp):
    """

    :param responder:
    :param dp:
    :return:
    """
    responses = {}
    for q, answers in reader.read(dp):
        response = responder(q)
        answers_text = [a[0] for a in answers]
        responses_to_comments = responder_multiple(answers_text)
        for a,rc in zip(answers, responses_to_comments):
            responses[a[2]] = response, a[0], rc
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
embedding_vectorizer = EmbeddingLookup("/home/martin/data/qatarliving/embeddings/qatarliving_qc_size100_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin")



def getSimilarities(responses, dp):
    similarities = defaultdict(dict)
    for q, answers in reader.read(dp):
        response = responses[answers[0][2]][0]
        responses_to_comments = [responses[a[2]][2] for a in answers]
        sims = computeSimilarities([a[0] for a in answers], response)
        addSimilarityFeatures(answers, sims, similarities, 'tfidf-cosine')
        sims2 = bm25.transform(response, [a[0] for a in answers])
        addSimilarityFeatures(answers, sims2, similarities, 'bm25')
        embedding_similarities = computeSimilarities([a[0] for a in answers], response,  embedding_vectorizer)
        addSimilarityFeatures(answers, embedding_similarities, similarities, 'embeddings')
        tfidf_generated_question_sims = computeSimilarities(responses_to_comments, q)
        addSimilarityFeatures(answers, tfidf_generated_question_sims, similarities, 'tfidf-cosine-generated-question')
        bm25_generated_question_sims = bm25.transform(q, responses_to_comments)
        addSimilarityFeatures(answers, bm25_generated_question_sims, similarities, 'bm25-generated-question')
        embedding_similarities_generated_question = computeSimilarities(responses_to_comments, q, embedding_vectorizer)
        addSimilarityFeatures(answers, embedding_similarities_generated_question, similarities, 'embeddings-generated-question')
    return similarities


def addSimilarityFeatures(answers, sims, similarities, label):
    sorted_by_sim = sorted(zip(answers, sims), key=lambda x: x[1], reverse=True)
    for i, (a, sim) in enumerate(sorted_by_sim):
        similarities[a[2]][label] = sim,
        similarities[a[2]]["%s-rank" % label] = i


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
                   "tfidf-cosine-rank",
                   "bm25",
                   "bm25-rank",
                   'embeddings',
                   'embeddings-rank',
                   "tfidf-cosine-generated-question",
                   "tfidf-cosine-generated-question-rank",
                   "bm25-generated-question",
                   "bm25-generated-question-rank",
                   "embeddings-generated-question",
                   "embeddings-generated-question-rank"]

feature_combinations = [
    ["tfidf-cosine",
     "tfidf-cosine-rank"],
    ["bm25",
     "bm25-rank"],
    ['embeddings',
     'embeddings-rank'],
    ["tfidf-cosine-generated-question",
     "tfidf-cosine-generated-question-rank"],
    ["bm25-generated-question",
     "bm25-generated-question-rank"],
    ["embeddings-generated-question",
     "embeddings-generated-question-rank"]
]


with tf.Session() as sess:
    responder, responder_multiple = decode(sess)
    for config in configs:
        print("Processing %s" % config['xml'])
        responses = generateHypothesisReferences(responder, responder_multiple, config['xml'])
        print("Hypotheses generated %s" % config['xml'])
        hyp_file, ref_file, order = persistResponses(responses, config['responsesFile'])
        print("Hypotheses persisted %s" % config['xml'])
        similarity_features = getSimilarities(responses, config['xml'])
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
