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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nltk.translate import bleu
import data_utils_udc
import seq2seq_model
import QLXMLReaderPy
from sklearn.metrics.pairwise import cosine_distances



tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("eval", False,
                            "Set to True for evaluation")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("dataset_type", "udc", "udc or ql")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils_udc.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  print("Preparing data in %s" % FLAGS.data_dir)
  questions_train, answers_train, questions_dev, answers_dev, _, _, _,  = data_utils_udc.prepare_data(
      FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.dataset_type)
  print("reading dictionaries")
  vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.qa" % FLAGS.en_vocab_size)
  vocab, rev_vocab = data_utils_udc.initialize_vocabulary(vocab_path)
  # Read data into buckets and compute their sizes.
  print ("Reading development and training data (limit: %d)."
         % FLAGS.max_train_data_size)
  dev_set = read_data(questions_dev, answers_dev)
  train_set = read_data(questions_train, answers_train, FLAGS.max_train_data_size)
  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
  train_total_size = float(sum(train_bucket_sizes))

  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]
  vectorizer = data_utils_udc.tfidfVectorizer(FLAGS.data_dir, vocab, FLAGS.dataset_type)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)
    
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
         train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        MAP = evaluateMAPParameterized(sess, model, vocab, rev_vocab, vectorizer)

        print("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f loss %.2f %.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity, loss, MAP))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        if current_step % (5 * FLAGS.steps_per_checkpoint) == 0:
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)

          _, eval_loss, outputs = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          total_bleu = 0.0
          total_docs = 0
          with open(os.path.join(FLAGS.data_dir,'evolution/responseEvolution-%d' % (model.global_step.eval())), 'w') as out:
            out.write('Global step %d perplexity %.2f responses: +\n' % (model.global_step.eval(), perplexity))
            vocab_outcomes = set()
            vocab_expected = set()
            bleu_score = 0.0
            outcomes_T = np.array([np.argmax(logit, axis=1) for logit in outputs]).T
            inputs_T = np.array(encoder_inputs).T
            expected_T = np.array(decoder_inputs).T
            for ex in range(len(outcomes_T)):
              ut = [rev_vocab[w] for w in inputs_T[ex] if w != 0]
              ut.reverse()
              current_outcome = outcomes_T[ex].tolist()
              if data_utils_udc.EOS_ID in current_outcome:
                current_outcome = current_outcome[:current_outcome.index(data_utils_udc.EOS_ID)]
              response = [rev_vocab[w] for w in current_outcome if w != 0]
              expected = [rev_vocab[w] for w in expected_T[ex] if w != 0]
              try:
                  bleu_score += bleu([expected[1:-1]], response, [1])
              except:
                  print("Error computing BLEU", expected[1:-1], response)
              vocab_outcomes = vocab_outcomes.union(set(response))
              vocab_expected = vocab_expected.union(set(expected))
              out.write("\t".join(["Utterance:", " ".join(ut)]) + "\n")
              out.write("\t".join(["Response:", " ".join(response)]) + "\n")
              out.write("\t".join(["Expected:", " ".join(expected)]) + "\n")
            print("  vocab_outcomes size: %d vocab_expected size %d overlap size: %d BLEU: %.2f" %
                  (len(vocab_outcomes), len(vocab_expected), len(vocab_outcomes.intersection(vocab_expected)), bleu_score/len(outcomes_T)))
            out.write("  vocab_outcomes size: %d vocab_expected size %d overlap size: %d BLEU: %.2f" %
                  (len(vocab_outcomes), len(vocab_expected), len(vocab_outcomes.intersection(vocab_expected)), bleu_score/len(outcomes_T)))
            total_bleu += bleu_score
            total_docs += len(outcomes_T)
        BLEU = total_bleu / total_docs
        if MAP > model.best_map.eval():
            print("MAP increased from %.2f to %.2f" % (model.best_map.eval(), MAP))
            sess.run(model.best_map.assign(MAP))
            model.map_saver.save(sess, FLAGS.train_dir+"/best-map/", global_step=model.global_step.eval())
        if BLEU > model.best_bleu.eval():
            print("BLEU increased from %.2f to %.2f" % (model.best_bleu.eval(), BLEU))
            sess.run(model.best_bleu.assign(BLEU))
            model.map_saver.save(sess, FLAGS.train_dir+"/best-bleu/", global_step=model.global_step.eval())
        with open(os.path.join(FLAGS.train_dir, 'scoreEvolution'), 'a') as out:
            print("  step: %d perplexity: %.4f MAP: %.4f BLEU: %.4f" %
                  (model.global_step.eval(), perplexity, MAP, total_bleu/total_docs))
            out.write("\t".join(str(x) for x in [model.global_step.eval(), perplexity, MAP, total_bleu / total_docs])+ "\n")
        sys.stdout.flush()


def decode(sess):
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.
    vocab, rev_vocab = getVocabularies()

    def responder(sentence):
        return evalSentence(sentence,model, vocab, rev_vocab, sess)
    return responder



def getVocabularies():
    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                              "vocab%d.qa" % FLAGS.en_vocab_size)
    return data_utils_udc.initialize_vocabulary(vocab_path)


def evalSentence(sentence, model, en_vocab, rev_fr_vocab, sess):
  # Get token-ids for the input sentence.
  token_ids = data_utils_udc.sentence_to_token_ids(sentence, en_vocab)
  if (token_ids is None):
      return ""
  # Which bucket does it belong to?
  bucket_id = len(_buckets) - 1
  for i, bucket in enumerate(_buckets):
    if bucket[0] >= len(token_ids):
      bucket_id = i
      break
  else:
    logging.warning("Sentence truncated: %s", sentence) 

  old_batch_size = model.batch_size
  model.batch_size = 1
  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)
  # Get output logits for the sentence.
  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  model.batch_size = old_batch_size
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils_udc.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils_udc.EOS_ID)]
  # Print out French sentence corresponding to outputs.
  return " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])


def evaluateMAP():
    vocab, rev_vocab = getVocabularies()
    vectorizer = data_utils_udc.tfidfVectorizer(FLAGS.data_dir, vocab, FLAGS.dataset_type)

    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.
        return evaluateMAPParameterized(sess, model, vocab, rev_vocab, vectorizer)


def evaluateMAPParameterized(sess, model, vocab, rev_vocab, vectorizer):
    meanAvgPrecision = 0.0
    meanAvgPrecisionQ = 0.0
    totalQueries = 0
    for (q, answers) in QLXMLReaderPy.read(
            "/home/martin/data/qatarliving/dev/SemEval2016-Task3-CQA-QL-dev-subtaskA-with-multiline.xml"):
        answers_transformed = vectorizer.transform([a[0] for a in answers])
        correct = [a[1] for a in answers]
        if sum(correct) == 0:
            continue
        response = evalSentence(q, model, vocab, rev_vocab, sess)
        response_transformed = vectorizer.transform([response])
        score = calculateMAP(answers_transformed, correct, response_transformed)
        question_transformed = vectorizer.transform([q])
        score_q = calculateMAP(answers_transformed, correct, question_transformed)
        meanAvgPrecision += score
        meanAvgPrecisionQ += score_q
        totalQueries += 1
    meanAvgPrecision /= totalQueries
    meanAvgPrecisionQ /= totalQueries
    print("MAP:", meanAvgPrecision)
    print("MAP Question:", meanAvgPrecisionQ)
    return meanAvgPrecision


def calculateMAP(answers_transformed, correct, response_transformed):
    distances = cosine_distances(response_transformed, answers_transformed)
    results = list(zip(distances[0], correct))
    results.sort()
    relevant_docs = 0
    score = 0.0
    for i, r in enumerate(results):
        if r[1]:
            relevant_docs += 1
            score += relevant_docs / (i + 1)
    score = score / relevant_docs
    return score


def main(_):
  if FLAGS.decode:
      with tf.Session() as sess:
          responder = decode(sess)
          # Decode from standard input.
          sys.stdout.write("> ")
          sys.stdout.flush()
          sentence = sys.stdin.readline()
          while sentence:
              response = responder(sentence)
              print(response)
              print("> ", end="")
              sys.stdout.flush()
              sentence = sys.stdin.readline()
  elif FLAGS.eval:
    evaluateMAP()
  else:
    train()





if __name__ == "__main__":
  tf.app.run()
