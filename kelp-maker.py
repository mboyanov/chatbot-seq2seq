from chatbot import decode
import tensorflow as tf
import QLXMLReaderPy as reader
import data_utils_udc
from sklearn.metrics.pairwise import cosine_distances
import os

ql_home = "/home/martin/data/qatarliving"
dps = [
   # os.path.join(ql_home, 'train/SemEval2016-Task3-CQA-QL-train-part1-with-multiline.xml'),
    #   os.path.join(ql_home, 'train/SemEval2016-Task3-CQA-QL-train-part2-with-multiline.xml'),
    os.path.join(ql_home, 'test/SemEval2016-Task3-CQA-QL-test-subtaskA-with-multiline.xml'),
    os.path.join(ql_home, 'dev/SemEval2016-Task3-CQA-QL-dev-subtaskA-with-multiline.xml')]
data_dir = os.path.join(ql_home, 'matchedPairs_ver5')
vocab_path = os.path.join(data_dir, 'vocab40000.qa')

vocab = data_utils_udc.initialize_tokenizer(vocab_path, False).vocab
default_vectorizer = data_utils_udc.tfidfVectorizer(data_dir, vocab, 'ql')

def getSimilarity(a, response, vectorizer=default_vectorizer):
    response_transformed = vectorizer.transform([response])
    answers_transformed = vectorizer.transform([a])
    distances = cosine_distances(response_transformed, answers_transformed)
    return distances[0]



def getSimilarities(responder):
    similarities = {}
    for dp in dps:
        for q,answers in reader.read(dp):
            response = responder(q)
            for a in answers:
                sim = getSimilarity(a[0], response)
                similarities[a[2]] = sim
    return similarities

sentence = "Good |<||BT:tree|(ROOT (S (NP (NN (massage::n))(NN (oil::n)))(VP (VBZ (be::v)))(ADVP (RB (there::r)))(NP (DT (any::d))(NN (place::n)))(NP (PRP (i::p)))(VP (MD (can::m))(VB (find::v)))(NP (VBN (scent::v))(NN (massage::n))(NNS (oils::n)))(PP (IN (in::i)))(NP (NN (qatar::n)))))|ET||BS:text|massage oil is there any place i can find scented massage oils in qatar?|ES||,||BT:tree|(ROOT (S (VP (VB (try::v)))(NP (DT (both::d)))(NP (-RRB- (}::-))(NNP (i'am::n)))(VP (RB (just::r))(VBG (try::v))(TO (to::t))(VB (be::v)))(ADJP (JJ (helpful::j))))(S (PP (IN (on::i)))(NP (DT (a::d))(JJ (serious::j))(NN (note::n)))(VP (VB (please::v))(VB (go::v)))(ADVP (RB (there::r))))(S (NP (PRP (you::p)))(VP (MD ('ll::m))(VB (find::v)))(NP (WP (what::w)))(NP (PRP (you::p)))(VP (VBP (be::v))(VBG (look::v)))(PP (IN (for::i)))))|ET||BS:text|Try Both ;) I'am just trying to be helpful. On a serious note - Please go there. you'll find what you are looking for.|ES||>| |BDV:WSsim|0.0 0.8005682229995728 0.08966762572526932 0.7889324426651001 0.5595598816871643 0.6260497570037842 |EDV||BDV:features|0.303022 0.172976 0.123091 0.000000 0.000000 0.000000 0.285124 0.237289 0.053376 0.267606 0.387435 0.521127 0.073298 0.085714 0.000000 0.000000 0.000000 0.214286 0.000000 0.151511 0.579746 |EDV||BV:threadFeats|MULT_MID:1.000000 POSITION:0.200000 DIAL_Uq_IN:1.000000 CATEGORY_Qatar_Living_Lounge:1.000000 MULT_REAL:0.200000 LENGTH:0.295000 MULT_BOOL:1.000000 |EV||BS:info|Q1_R1_C5|ES|"
sentence_plus = "Good |<||BT:tree|(ROOT (S (NP (NN (massage::n))(NN (oil::n)))(VP (VBZ (be::v)))(ADVP (RB (there::r)))(NP (DT (any::d))(NN (place::n)))(NP (PRP (i::p)))(VP (MD (can::m))(VB (find::v)))(NP (VBN (scent::v))(NN (massage::n))(NNS (oils::n)))(PP (IN (in::i)))(NP (NN (qatar::n)))))|ET||BS:text|massage oil is there any place i can find scented massage oils in qatar?|ES||,||BT:tree|(ROOT (S (VP (VB (try::v)))(NP (DT (both::d)))(NP (-RRB- (}::-))(NNP (i'am::n)))(VP (RB (just::r))(VBG (try::v))(TO (to::t))(VB (be::v)))(ADJP (JJ (helpful::j))))(S (PP (IN (on::i)))(NP (DT (a::d))(JJ (serious::j))(NN (note::n)))(VP (VB (please::v))(VB (go::v)))(ADVP (RB (there::r))))(S (NP (PRP (you::p)))(VP (MD ('ll::m))(VB (find::v)))(NP (WP (what::w)))(NP (PRP (you::p)))(VP (VBP (be::v))(VBG (look::v)))(PP (IN (for::i)))))|ET||BS:text|Try Both ;) I'am just trying to be helpful. On a serious note - Please go there. you'll find what you are looking for.|ES||>| |BDV:WSsim|0.0 0.8005682229995728 0.08966762572526932 0.7889324426651001 0.5595598816871643 0.6260497570037842 |EDV||BDV:features|0.303022 0.172976 0.123091 0.000000 0.000000 0.000000 0.285124 0.237289 0.053376 0.267606 0.387435 0.521127 0.073298 0.085714 0.000000 0.000000 0.000000 0.214286 0.000000 0.151511 0.579746 1.000000 |EDV||BV:threadFeats|MULT_MID:1.000000 POSITION:0.200000 DIAL_Uq_IN:1.000000 CATEGORY_Qatar_Living_Lounge:1.000000 MULT_REAL:0.200000 LENGTH:0.295000 MULT_BOOL:1.000000 |EV||BS:info|Q1_R1_C5|ES|"
import re
id_regex = r'BS:info\|([^\|]+)\|'
text_regex = r'BS:text\|([^\|]+)\|'
feature_regex = r'BDV:features\|([^\|]+)\|'
def find_id(s):
    matches = re.findall(id_regex, s)
    return matches[0]

def add_feature(feature, s):
    features = re.findall(feature_regex, s)[0]
    features += "%.6f " % feature
    return re.sub(feature_regex, 'BDV:features|'+features+"|", s)

def find_text(s):
    return re.findall(text_regex, s)

assert(find_id(sentence)=="Q1_R1_C5")
assert(add_feature(1.0, sentence) == sentence_plus)
kelp_in_path = '/home/martin/projects/tutorial/SemEval2016-Task3-CQA-QL-dev-with-multiline.xml.taskA.klp'
kelp_out_path = '/home/martin/projects/tutorial/SemEval2016-Task3-CQA-QL-dev-with-multiline-with-bot.xml.taskA.klp'
with tf.Session() as sess, open(kelp_in_path) as kelp_in, open(kelp_out_path,'w') as kelp_out:
    responder = decode(sess)
    similarities = getSimilarities(responder)
    for line in kelp_in:
        c_id = find_id(line)
        if c_id not in similarities:
            print('Sim is none for %s ' % c_id)
            continue
        sim = similarities[c_id]
        line_plus = add_feature(sim, line)
        kelp_out.write(line_plus)


