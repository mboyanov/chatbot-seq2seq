import tensorflow as tf
from tensorflow.python.ops import embedding_ops
import numpy as np

import data_utils_udc
default_tokenizer = data_utils_udc.bpe_tokenizer


class Seq2SeqEmbeddingLookup():

    def __init__(self, session, tokenizer=default_tokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def transform(self, X):
        inputs = [self.tokenizer.transform(x) for x in X]
        with tf.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
            embeddings = tf.get_variable('embedding', shape=(40257, 256))
            embedded = embedding_ops.embedding_lookup(
                embeddings,inputs)
            embedded_output = self.session.run(embedded)
            embedded_output_centroids = np.sum(embedded_output, axis= 2)
            return embedded_output_centroids

