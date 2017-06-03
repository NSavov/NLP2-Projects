import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data
from neuralibm1 import NeuralIBM1Model

# for TF 1.1
try:
  from tensorflow.contrib.keras.initializers import glorot_uniform
except:  # for TF 1.0
  from tensorflow.contrib.layers import xavier_initializer as glorot_uniform


class NeuralIBM1ModelContext(NeuralIBM1Model):
    "our neural ibm model 1 with additional french context"

    def __init__(self, batch_size=8,
               x_vocabulary=None, y_vocabulary=None,
               emb_dim=32, mlp_dim=64,
               session=None):
        super().__init__(batch_size, x_vocabulary, y_vocabulary, emb_dim, mlp_dim, session)

    def _create_weights(self):
        """Create weights for the model."""
        with tf.variable_scope("MLP") as scope:
            self.mlp_W_ = tf.get_variable(
                name="W_", initializer=glorot_uniform(),
                shape=[(self.emb_dim * 2), self.mlp_dim])

            self.mlp_b_ = tf.get_variable(
                name="b_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_W = tf.get_variable(
                name="W", initializer=glorot_uniform(),
                shape=[self.mlp_dim, self.y_vocabulary_size])

            self.mlp_b = tf.get_variable(
                name="b", initializer=tf.constant_initializer(0.0),
                shape=[self.y_vocabulary_size])

    def _build_model(self):
        """Builds the computational graph for our model."""

        # first we need to know some sizes from the current input data
        batch_size = tf.shape(self.x)[0]
        longest_x = tf.shape(self.x)[1]  # longest M
        longest_y = tf.shape(self.y)[1]  # longest N

        # 1. Let's create a (source) word embeddings matrix.
        # These are trainable parameters, so we use tf.get_variable.
        # Shape: [Vx, emb_dim] where Vx is the source vocabulary size
        x_embeddings = tf.get_variable(
            name="x_embeddings", initializer=tf.random_uniform_initializer(),
            shape=[self.x_vocabulary_size, self.emb_dim])

        # create the (target) word embeddings matrix.
        # shape: [Vy, emb_dim]
        y_embeddings = tf.get_variable(
            name="y_embeddings", initializer=tf.random_uniform_initializer(),
            shape=[self.y_vocabulary_size, self.emb_dim])

        # Now we start defining our graph.

        # This looks up the embedding vector for each word given the word IDs in self.x.
        # Shape: [B, M, emb_dim] where B is batch size, M is (longest) source sentence length.
        x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x)

        # This looks up the embedding vector for each word given the word IDs in self.y.
        # Shape: [B, N, emb_dim] where B is batch size, N is (longest) target sentence length.
        y_prev = tf.multiply(tf.ones([batch_size, 1], dtype="int64"), 2)
        y_padded = tf.concat([y_prev, self.y], 1)
        y_padded = y_padded[:, 0:-1]
        y_embedded = tf.nn.embedding_lookup(y_embeddings, y_padded)

        # 2. Now we define the generative model P(Y | X=x)

        # It's also useful to have masks that indicate what
        # values of our batch we should ignore.
        # Masks have the same shape as our inputs, and contain
        # 1.0 where there is a value, and 0.0 where there is padding.
        x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
        y_mask = tf.cast(tf.sign(self.y), tf.float32)  # Shape: [B, N]
        x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
        y_len = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]

        # 2.a Build an alignment model P(A | X, M, N)

        # This just gives you 1/length_x (already including NULL) per sample.
        # i.e. the lengths are the same for each word y_1 .. y_N.
        lengths = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
        pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))  # Shape: [B, M]

        # We now have a matrix with 1/M values.
        # For a batch of 2 sentences, with lengths 2 and 3:
        #
        #  pa_x = [[1/2 1/2   0]
        #          [1/3 1/3 1/3]]
        #
        # But later we will need it N times. So we repeat (=tile) this
        # matrix N times, and for that we create a new dimension
        # in between the current ones (dimension 1).
        pa_x = tf.expand_dims(pa_x, 1)  # Shape: [B, 1, M]

        #  pa_x = [[[1/2 1/2   0]]
        #          [[1/3 1/3 1/3]]]
        # Note the extra brackets.

        # Now we perform the tiling:
        pa_x = tf.tile(pa_x, [1, longest_y, 1])  # [B, N, M]

        # Result:
        #  pa_x = [[[1/2 1/2   0]
        #           [1/2 1/2   0]]
        #           [[1/3 1/3 1/3]
        #           [1/3 1/3 1/3]]]

        # 2.b append the french and english embeddings

        # Copy every y_embedding in every sentence M times, and reshape to correct dimensions
        # Shape: [B, N * M, emb_dim]
        # Result for one sentence:
        # y_embedded =[[   [le]   [le]   [le]
        #               [chien][chien][chien]
        #                [noir] [noir] [noir]]]
        y_embedded = tf.tile(y_embedded, [1, 1, longest_x])
        y_embedded = tf.transpose(y_embedded, [1, 0, 2])
        y_embedded = tf.reshape(y_embedded, [batch_size, longest_x * longest_y, self.emb_dim])

        # now copy every x_embedded sentence N times
        # Shape: [B, N * M, emb_dim]
        # Result for one sentence:
        # x_embedded =[[[null][black][dog]
        #               [null][black][dog]
        #               [null][black][dog]]]
        x_embedded = tf.tile(x_embedded, [1, longest_y, 1])

        # Finally, append the french and english embeddings
        embedded = tf.concat([y_embedded, x_embedded], 2)

        # 2.c P(Y | X, A) = P(Y | X_A)

        # Then we make the input to the MLP 2-D.
        # Every output row will be of size Vy, and after a softmax
        # will sum to 1.0.
        mlp_input = tf.reshape(embedded, [batch_size * longest_x * longest_y, self.emb_dim * 2])

        # Here we apply the MLP to our input.
        h = tf.matmul(mlp_input, self.mlp_W_) + self.mlp_b_  # affine transformation
        h = tf.tanh(h)  # non-linearity
        h = tf.matmul(h, self.mlp_W) + self.mlp_b  # affine transformation [B * M, Vy]

        # You could also use TF fully connected to create the MLP.
        # Then you don't have to specify all the weights and biases separately.
        # h = tf.contrib.layers.fully_connected(mlp_input, self.mlp_dim, activation_fn=tf.tanh, trainable=True)
        # h = tf.contrib.layers.fully_connected(h, self.y_vocabulary_size, activation_fn=None, trainable=True)

        # Now we perform a softmax which operates on a per-row basis.
        py_xa = tf.nn.softmax(h)
        py_xa = tf.reshape(py_xa, [batch_size, longest_y, longest_x, self.y_vocabulary_size])

        # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)

        # Here comes a rather fancy matrix multiplication.
        # Note that tf.matmul is defined to do a matrix multiplication
        # [N, M] @ [M, Vy] for each item in the first dimension B.
        # So in the final result we have B matrices [N, Vy], i.e. [B, N, Vy].
        #
        # We matrix-multiply:
        #   pa_x       Shape: [B, N, 1, *M*]
        # and
        #   py_xa      Shape: [B, N, *M*, Vy]
        # to get
        #   py_x  Shape: [B, N, 1, Vy]
        #
        # Which is simply one probability vector for every french position, which we require
        #
        # Note: P(y|x) = prod_j p(y_j|x) = prod_j sum_aj p(aj|m)p(y_j|x_aj)
        #
        pa_x = tf.expand_dims(pa_x, 2)  # Shape: [B, N, 1, M]
        py_x = tf.matmul(pa_x, py_xa)  # Shape: [B, N, 1, Vy]
        py_x = tf.reshape(py_x, [batch_size, longest_y, self.y_vocabulary_size])  # shape: [B, N, Vy]

        # This calculates the accuracy, i.e. how many predictions we got right.
        predictions = tf.argmax(py_x, axis=2)
        acc = tf.equal(predictions, self.y)
        acc = tf.cast(acc, tf.float32) * y_mask
        acc_correct = tf.reduce_sum(acc)
        acc_total = tf.reduce_sum(y_mask)
        acc = acc_correct / acc_total

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.y, [-1]),
            logits=tf.log(tf.reshape(py_x, [batch_size * longest_y, self.y_vocabulary_size])),
            name="logits"
        )
        cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
        cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
        cross_entropy = tf.reduce_mean(cross_entropy, axis=0)

        # Now we define our cross entropy loss
        # Play with this if you want to try and replace TensorFlow's CE function.
        # Disclaimer: untested code
        #     y_one_hot = tf.one_hot(self.y, depth=self.y_vocabulary_size)     # [B, N, Vy]
        #     cross_entropy = tf.reduce_sum(y_one_hot * tf.log(py_x), axis=2)  # [B, N]
        #     cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)    # [B]
        #     cross_entropy = -tf.reduce_mean(cross_entropy)  # scalar

        self.pa_x = pa_x
        self.py_x = py_x
        self.py_xa = py_xa
        self.loss = cross_entropy
        self.predictions = predictions
        self.accuracy = acc
        self.accuracy_correct = tf.cast(acc_correct, tf.int64)
        self.accuracy_total = tf.cast(acc_total, tf.int64)

    def get_viterbi(self, x, y):
        """Returns the Viterbi alignment for (x, y)"""

        feed_dict = {
            self.x: x,  # English
            self.y: y  # French
        }

        # run model on this input
        py_xa, acc_correct, acc_total = self.session.run(
            [self.py_xa, self.accuracy_correct, self.accuracy_total],
            feed_dict=feed_dict)

        # things to return
        batch_size, longest_y = y.shape
        alignments = np.zeros((batch_size, longest_y), dtype="int64")
        probabilities = np.zeros((batch_size, longest_y), dtype="float32")

        for b, sentence in enumerate(y):
            for j, french_word in enumerate(sentence):
                if french_word == 0:  # Padding
                    break

                # prob for sentence b, prev french type j, all english alignments, for word y at pos j in sen b
                probs = py_xa[b, j, :, y[b, j]]

                # find the max alignment and alignment prob
                a_j = probs.argmax()
                p_j = probs[a_j]

                # store
                alignments[b, j] = a_j
                probabilities[b, j] = p_j

        return alignments, probabilities, acc_correct, acc_total



