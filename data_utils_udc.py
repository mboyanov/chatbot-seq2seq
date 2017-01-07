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
import re

from tensorflow.python.platform import gfile
from collections import defaultdict
from UDCDatasetReader import UDCDatasetReader
from QLDatasetReader import QLDatasetReader

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

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
_DIGIT_RE_B = re.compile(br"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        if type(space_separated_fragment) == 'str':
            words.extend(re.split("([.,!?\"':;)(])", space_separated_fragment))
        else:
            words.extend(re.split("([.,!?\"':;)(])", space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, dataset_reader,
                      tokenizer=basic_tokenizer, normalize_digits=True):
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
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = defaultdict(lambda: 0)
        for w in dataset_reader.words(data_path, tokenizer):
            word = _DIGIT_RE.sub("0", w) if normalize_digits else w
            vocab[word] += 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print('Total words', len(vocab_list))
        print('top20', vocab_list[:20])
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(str(w) + "\n")


def initialize_vocabulary(vocabulary_path):
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
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    if len(words) == 0:
        return None
    if type(words[0] == 'str'):
        return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]
    else:
        return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, questions_path, answers_path, vocabulary_path, dataset_reader,
                      tokenizer=basic_tokenizer, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if gfile.Exists(questions_path) and gfile.Exists(answers_path):
        print("files already tokenized")
        return
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(questions_path, mode="w") as questions_tokens_file:
        with gfile.GFile(answers_path, mode="w") as answers_tokens_file:
            for q, a in dataset_reader.conversations(data_path, tokenizer):
                token_ids = sentence_to_token_ids(q, vocab, tokenizer, normalize_digits)
                token_ids_answer = sentence_to_token_ids(a, vocab, tokenizer, normalize_digits)
                if (token_ids is not None and token_ids_answer is not None):
                    questions_tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    answers_tokens_file.write(" ".join([str(tok) for tok in token_ids_answer]) + "\n")



def prepare_data(data_dir, en_vocabulary_size, fr_vocabulary_size, dataset_type, tokenizer=basic_tokenizer):
    """Get UDC data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      en_vocabulary_size: size of the English vocabulary to create and use.
      fr_vocabulary_size: size of the French vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for English training data-set,
        (2) path to the token-ids for French training data-set,
        (3) path to the token-ids for English development data-set,
        (4) path to the token-ids for French development data-set,
        (5) path to the English vocabulary file,
        (6) path to the French vocabulary file.
    """

    # Create vocabularies of the appropriate sizes.
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    train_dataset_reader, test_dataset_reader = getReadersByDatasetType(dataset_type)
    vocab_path = os.path.join(data_dir, "vocab%d.qa" % fr_vocabulary_size)
    create_vocabulary(vocab_path, train_path, fr_vocabulary_size, train_dataset_reader, tokenizer)

    # Create token ids for the training data.
    question_train_ids_path = data_dir + ("question.ids%d.train" % fr_vocabulary_size)
    answer_train_ids_path = data_dir + ("answer.ids%d.train" % en_vocabulary_size)
    question_test_ids_path = data_dir + ("question.ids%d.test" % fr_vocabulary_size)
    answer_test_ids_path = data_dir + ("answer.ids%d.test" % en_vocabulary_size)
    data_to_token_ids(train_path, question_train_ids_path, answer_train_ids_path,
                      vocab_path,
                      train_dataset_reader,
                      tokenizer)
    data_to_token_ids(test_path, question_test_ids_path, answer_test_ids_path,
                      vocab_path,
                      test_dataset_reader,
                      tokenizer)

    return (question_train_ids_path, answer_train_ids_path,
            question_test_ids_path, answer_test_ids_path,
            vocab_path, vocab_path)

def getReadersByDatasetType(dataset_type):
    if (dataset_type == 'udc'):
        return UDCDatasetReader(True), UDCDatasetReader(False)
    else:
        qlReader = QLDatasetReader()
        return qlReader, qlReader