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
import sys
import time

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import  sentence_bleu as bleu
from six.moves import xrange  # pylint: disable=redefined-builtin

import QLXMLReaderPy
import data_utils_udc
import explorer
import progress_bar
import seq2seq_model
from evaluators.bleu_evaluator import BLEUEvaluator
from evaluators.map_evaluator import MAPEvaluator
from evaluators.persister_evaluator import PersisterEvaluator
from evaluators.vocabulary_evaluator import VocabularyEvaluator
from evaluators.map_evaluator_summed import MAPEvaluatorSummed
from evaluators.ttr_evaluator import TTREvaluator
from evaluators.length_evaluator import LengthEvaluator
import trainingFilesReader

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
tf.app.flags.DEFINE_boolean("explore", False,
                            "run exploration statistics")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("dataset_type", "udc", "udc or ql")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40,45)]





def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size + 257,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype,
      dropout=0.3)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  print("Model initialized")
  return model

from execution_plan import ExecutionPlan

def train():
    execution_plan = ExecutionPlan('/home/martin/data/udc/', os.path.join(ql_home, 'matchedPairs_ver5/'), 50000, 50000, FLAGS.en_vocab_size, _buckets)
    questions_train, answers_train, questions_dev, answers_dev, _, _, _, = data_utils_udc.prepare_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.dataset_type)
    print("reading dictionaries")
    vocab_path = os.path.join(FLAGS.data_dir,
                              "vocab%d.qa" % FLAGS.en_vocab_size)
    tokenizer = data_utils_udc.initialize_tokenizer(vocab_path)
    normal_tokenizer = data_utils_udc.initialize_tokenizer(vocab_path, is_bpe=False)
    vectorizer = data_utils_udc.tfidfVectorizer(FLAGS.data_dir, normal_tokenizer.vocab, FLAGS.dataset_type)
    print("Preparing data in %s" % FLAGS.data_dir)

    # Read data into buckets and compute their sizes.
    print("Reading development and training data (limit: %d)."
          % FLAGS.max_train_data_size)
    while True:
        trainABit(execution_plan)
        tf.reset_default_graph()
        evaluate(tokenizer, vectorizer)
        tf.reset_default_graph()

def trainABit(execution_plan):
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)
    dev_set, train_set = execution_plan.getData(model.global_step.eval())
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    print("Train bucket sizes", train_bucket_sizes)
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while current_step <= 10 * FLAGS.steps_per_checkpoint:
      progress_bar.printProgressBar(current_step, 10 * FLAGS.steps_per_checkpoint)
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
        print("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f loss %.2f " % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity, loss))
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
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)

          _, eval_loss, outputs = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

        sys.stdout.flush()

def evaluate(tokenizer = None, vectorizer = None):
    with tf.Session() as sess:
        model = create_model(sess, True)
        if tokenizer is None:
            vocab_path = os.path.join(FLAGS.data_dir,
                                      "vocab%d.qa" % FLAGS.en_vocab_size)
            tokenizer = data_utils_udc.initialize_tokenizer(vocab_path)
            tokenizer_normal = data_utils_udc.initialize_tokenizer(vocab_path, False)
        if vectorizer is None:
            vectorizer = data_utils_udc.tfidfVectorizer(FLAGS.data_dir, tokenizer_normal.vocab, FLAGS.dataset_type)

        evaluateDataset(model, sess, tokenizer, vectorizer, 'test')
        MAP_dev, BLEU_dev = evaluateDataset(model, sess, tokenizer, vectorizer, 'dev')
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")

        if MAP_dev > model.best_map.eval():
            print("MAP increased from %.2f to %.2f" % (model.best_map.eval(), MAP_dev))
            sess.run(model.best_map.assign(MAP_dev))
            model.map_saver.save(sess, FLAGS.train_dir+"/best-map/", global_step=model.global_step.eval())
            model.saver.save(sess, checkpoint_path, global_step=model.global_step.eval())
        if BLEU_dev > model.best_bleu.eval():
            print("BLEU increased from %.2f to %.2f" % (model.best_bleu.eval(), BLEU_dev))
            sess.run(model.best_bleu.assign(BLEU_dev))
            model.map_saver.save(sess, FLAGS.train_dir+"/best-bleu/", global_step=model.global_step.eval())
            model.saver.save(sess, checkpoint_path, global_step=model.global_step.eval())
        print("  step: %d perplexity: MAP dev: %.4f BLEU: %.4f" %
              (model.global_step.eval(), MAP_dev, BLEU_dev))


def evaluateDataset(model, sess, tokenizer, vectorizer, ds):
    evaluators = [BLEUEvaluator(),
                  MAPEvaluator(),
                  PersisterEvaluator(
                      os.path.join(FLAGS.train_dir, 'responseEvolution-%s-%s' % (ds, model.global_step.eval()))),
                  VocabularyEvaluator(),
                  MAPEvaluatorSummed(),
                  LengthEvaluator(),
                  TTREvaluator()]
    visitDatasetParameterized(sess, model, tokenizer, vectorizer, ds, evaluators)
    MAP = evaluators[1].results()
    BLEU = evaluators[0].results()
    ## Persister evaluator saves in a different file, so just call results()
    evaluators[2].results()
    vocab_eval = evaluators[3].results()
    MAP_SUMMED = evaluators[4].results()
    LENGTH = evaluators[5].results()
    TTR = evaluators[6].results()
    score_path = os.path.join(FLAGS.train_dir, 'scoreEvolution-%s' % ds)
    if not os.path.isfile(score_path):
        with open(score_path, 'w') as out:
            out.write("\t".join(
                ["Global step", "MAP", "MAP_SUMMED", "BLEU", "Vocab size", "Target Vocab Size", "Intersection Vocab size", "LENGTH", "TTR"]) + "\n")
    with open(score_path, 'a') as out:
        out.write("\t".join([
            str(model.global_step.eval()), str(MAP), str(MAP_SUMMED), str(BLEU), str(vocab_eval[0][1]), str(vocab_eval[1][1]),
            str(vocab_eval[2][1]), str(LENGTH), str(TTR)
        ]) + "\n")
    return MAP, BLEU


def decode(sess):
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.
    tokenizer = getVocabularies()

    def responder(sentence):
        return evalSentence(sentence,model, tokenizer, sess)
    return responder



def getVocabularies():
    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir,
                              "vocab%d.qa" % FLAGS.en_vocab_size)
    return data_utils_udc.initialize_tokenizer(vocab_path)


def evalSentence(sentence, model, tokenizer, sess):
  # Get token-ids for the input sentence.
  token_ids = data_utils_udc.sentence_to_token_ids(sentence, tokenizer)
  if (token_ids is None):
      return ""
  # Which bucket does it belong to?
  bucket_id = len(_buckets) - 1
  for i, bucket in enumerate(_buckets):
    if bucket[0] >= len(token_ids):
      bucket_id = i
      break

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
  return tokenizer.inverse_transform(outputs)



ql_home = "/home/martin/data/qatarliving"
ql_sets = {}
ql_sets['dev'] = os.path.join(ql_home, "dev/SemEval2016-Task3-CQA-QL-dev-subtaskA-with-multiline.xml")
ql_sets['train'] = os.path.join(ql_home, "train/SemEval2016-Task3-CQA-QL-train-part1-subtaskA-with-multiline.xml")
ql_sets['test'] = os.path.join(ql_home, "test/SemEval2016-Task3-CQA-QL-test-subtaskA-with-multiline.xml")

def visitDatasetParameterized(sess, model, tokenizer, vectorizer, ds, evaluators = []):
    for (q, answers) in QLXMLReaderPy.read(ql_sets[ds]):
        correct = [a[1] for a in answers]
        if sum(correct) == 0:
            continue
        response = evalSentence(q, model, tokenizer, sess)
        for evaluator in evaluators:
            evaluator.update(q, response, answers, vectorizer)


def explore():
    explorer.explore([{
        'name': 'dev',
        'reader': QLXMLReaderPy.read(ql_sets["dev"])
    }, {
        'name': 'test',
        'reader' : QLXMLReaderPy.read(ql_sets['test'])
    }], data_utils_udc.bpe_tokenizer)
    questions_train, answers_train, _, _, _, _, _, = data_utils_udc.prepare_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.dataset_type)

    explorer.explore([{
        'name': 'train',
        'reader': trainingFilesReader.read(questions_train, answers_train)
    }], lambda x: x.split(), flat=True)



def main(_):
    if FLAGS.explore:
        explore()
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
        evaluate()
    else:
        train()





if __name__ == "__main__":
  tf.app.run()
