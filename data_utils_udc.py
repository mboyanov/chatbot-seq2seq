# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from UDCDatasetReader import UDCDatasetReader
from QLDatasetReader import QLDatasetReader, FilteredDatasetReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tokenizer.tokenizer import Tokenizer, BPETokenizer
import pickle
from bm25 import BM25

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

default_tokenizer = Tokenizer(_UNK)
bpe_tokenizer = BPETokenizer(open("/home/martin/projects/subword-nmt/vocab_bpe_merged"), _START_VOCAB)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, dataset_reader,
                      tokenizer=default_tokenizer, persist_counts = False):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      dataset_reader: used to read the dataset
      tokenizer: a function to use to tokenize each data sentence;
    """
    if not os.path.isfile(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab_list = tokenizer.fit(dataset_reader.conversations(data_path), max_vocabulary_size, _START_VOCAB)
        with open(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                if persist_counts:
                    vocab_file.write(str(w[0]) +" " + str(w[1]) + "\n")
                else:
                    vocab_file.write(str(w[0]) + "\n")


def initialize_tokenizer(vocabulary_path, is_bpe=True):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if is_bpe:
        return bpe_tokenizer
    if os.path.isfile(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        return Tokenizer(_UNK, vocab_list=rev_vocab)
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, tokenizer):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    words = tokenizer.transform(sentence)
    if len(words) == 0:
        return None
    return words


def data_to_token_ids(data_path, questions_path, answers_path, vocabulary_path, dataset_reader):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if os.path.isfile(questions_path) and os.path.isfile(answers_path):
        print("files already tokenized")
        return
    print("Tokenizing data in %s" % data_path)
    tokenizer = initialize_tokenizer(vocabulary_path)
    lengths_q = []
    lengths_a = []
    with open(questions_path, mode="w") as questions_tokens_file:
        with open(answers_path, mode="w") as answers_tokens_file:
            for q, a in dataset_reader.conversations(data_path):
                token_ids = sentence_to_token_ids(q, tokenizer)
                token_ids_answer = sentence_to_token_ids(a, tokenizer)
                if token_ids is not None and token_ids_answer is not None:
                    lengths_q.append(len(token_ids))
                    lengths_a.append(len(token_ids_answer))
                    questions_tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    answers_tokens_file.write(" ".join([str(tok) for tok in token_ids_answer]) + "\n")
    print(questions_path, 'stats', np.mean(lengths_q), np.std(lengths_q))
    print(answers_path, 'stats', np.mean(lengths_a), np.std(lengths_a))


def prepare_data(data_dir, vocabulary_size, dataset_type, tokenizer=default_tokenizer):
    """Get UDC data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      vocabulary_size: size of the English vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for questions training data-set,
        (2) path to the token-ids for answers training data-set,
        (3) path to the token-ids for questions development data-set,
        (4) path to the token-ids for answers development data-set,
        (5) path to the English vocabulary file,
        (6) path to the French vocabulary file.
    """

    # Create vocabularies of the appropriate sizes.
    train_path = os.path.join(data_dir, 'train.csv')
    train_dataset_reader, test_dataset_reader = getReadersByDatasetType(dataset_type)
    vocab_path = os.path.join(data_dir, "vocab%d.qa" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, vocabulary_size, train_dataset_reader, tokenizer)

    paths = []
    for (ds, reader) in [('train', train_dataset_reader), ('test', test_dataset_reader), ('validation', test_dataset_reader)]:
        path = os.path.join(data_dir, '%s.csv' % ds)
        question_ids_path = data_dir + ("question.ids%d.%s" % (vocabulary_size, ds))
        answer_ids_path = data_dir + ("answer.ids%d.%s" % (vocabulary_size, ds))
        paths.append(question_ids_path)
        paths.append(answer_ids_path)
        data_to_token_ids(path, question_ids_path, answer_ids_path,
                          vocab_path,
                          reader)
    paths.append(vocab_path)
    return paths

def tfidfVectorizer(data_dir, vocab, dataset_type, tokenizer=default_tokenizer.tokenize):
    train_path = os.path.join(data_dir, 'train.csv')
    if os.path.isfile(train_path+".vectorizer"):
        return pickle.load(open(train_path+".vectorizer", 'rb'))
    train_dataset_reader, _ = getReadersByDatasetType(dataset_type)
    answers = []
    for (q, a) in train_dataset_reader.conversations(train_path):
        answers.append(a)
    vectorizer = TfidfVectorizer(vocabulary=vocab, tokenizer=tokenizer)
    vectorizer.fit(answers)
    pickle.dump(vectorizer, open(train_path + ".vectorizer", 'wb'))
    return vectorizer



def bm25(data_dir, dataset_type, tokenizer=default_tokenizer):
    train_path = os.path.join(data_dir, 'train.csv')
    if os.path.isfile(train_path+".bm25"):
        bm25= pickle.load(open(train_path+".bm25", 'rb'))
        return bm25
    train_dataset_reader, _ = getReadersByDatasetType(dataset_type)
    answers = []
    for (q, a) in train_dataset_reader.conversations(train_path):
        answers.append(a)
    bm25 = BM25(tokenizer)
    bm25.fit(answers)
    pickle.dump(bm25, open(train_path + ".bm25", 'wb'))
    return bm25


def getReadersByDatasetType(dataset_type):
    if (dataset_type == 'udc'):
        return UDCDatasetReader(True), UDCDatasetReader(False)
    elif dataset_type == 'friends':
        from friendsHTMLReader import FriendsHTMLReader
        reader = FriendsHTMLReader()
        return reader, reader
    else:
        #filteredReader = FilteredDatasetReader()
        qlReader = QLDatasetReader()
        return qlReader, qlReader


# r1, r2 = getReadersByDatasetType('ql')
# r3, r4 = getReadersByDatasetType('udc')
#
#r1, r2 = getReadersByDatasetType('friends')
##
#create_vocabulary('/tmp/vocab_friends', '/mnt/8C24EDC524EDB1FE/data/friends/', 999999999, r1, persist_counts=True)
# create_vocabulary('/tmp/vocab_udc', '/home/martin/data/udc/train.csv', 999999999, r3, persist_counts=True)
