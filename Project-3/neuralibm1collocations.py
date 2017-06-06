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

class NeuralIBM1ModelCollocations(NeuralIBM1Model):
    "our neural ibm model 1 with additional french context"

    def __init__(self, batch_size=8,
               x_vocabulary=None, y_vocabulary=None,
               emb_dim=32, mlp_dim=64,
               session=None):
        super().__init__(batch_size, x_vocabulary, y_vocabulary, emb_dim, mlp_dim, session)


    def _create_weights(self):
        """Create weights for the model."""
        with tf.variable_scope("MLP") as scope:
            #Weights for P(C|F_prev=f)
            #input: d_F - embedding of f
            #output: 1 - the gate C - between 0 and 1 (not discrete yet)
            self.mlp_Wc_ = tf.get_variable(
                name="Wc_", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.mlp_dim])

            self.mlp_bc_ = tf.get_variable(
                name="bc_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Wc = tf.get_variable(
                name="Wc", initializer=glorot_uniform(),
                shape=[self.mlp_dim, 1])

            self.mlp_bc = tf.get_variable(
                name="bc", initializer=tf.constant_initializer(0.0),
                shape=[1])


            #Weights for P(F|F_prev=f)
            #input: d_F - embedding of f
            #output: V_F - softmax over all the french words
            self.mlp_Wf_ = tf.get_variable(
                name="Wf_", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.mlp_dim])

            self.mlp_bf_ = tf.get_variable(
                name="bf_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Wf = tf.get_variable(
                name="Wf", initializer=glorot_uniform(),
                shape=[self.mlp_dim, self.y_vocabulary_size])

            self.mlp_bf = tf.get_variable(
                name="bf", initializer=tf.constant_initializer(0.0),
                shape=[self.y_vocabulary_size])


            #Weights for P(F|E=e)
            #input: d_E - embedding of e
            #output: V_F - softmax over all the french words
            self.mlp_We_ = tf.get_variable(
                name="We_", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.mlp_dim])

            self.mlp_be_ = tf.get_variable(
                name="be_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_We = tf.get_variable(
                name="We", initializer=glorot_uniform(),
                shape=[self.mlp_dim, self.y_vocabulary_size])

            self.mlp_be = tf.get_variable(
                name="be", initializer=tf.constant_initializer(0.0),
                shape=[self.y_vocabulary_size])


    def _build_model(self):
        """Builds the computational graph for our model."""

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

        # 2. Now we define the generative model P(Y | X=x)

        # first we need to know some sizes from the current input data
        batch_size = tf.shape(self.x)[0]
        longest_x = tf.shape(self.x)[1]  # longest M
        longest_y = tf.shape(self.y)[1]  # longest N


        # This looks up the embedding vector for each word given the word IDs in self.y.
        # Shape: [B, N, emb_dim] where B is batch size, N is (longest) target sentence length.
        y_prev = tf.multiply(tf.ones([batch_size, 1], dtype="int64"), 2)
        y_padded = tf.concat([y_prev, self.y], 1)
        y_padded = y_padded[:, 0:-1]
        y_embedded = tf.nn.embedding_lookup(y_embeddings, y_padded)

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
        # For a batch of 2 setencnes, with lengths 2 and 3:
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

        # 2.b P(Y | X, A) = P(Y | X_A)

        # First we make the input to the MLP 2-D for C.
        # Every output row will be of size Vy, and after a sigmoid
        # we will get non-discrete version of C.

        c_input = tf.reshape(y_embedded, [batch_size * longest_y, self.emb_dim])

        # Here we apply the MLP to our input.
        hc = tf.matmul(c_input, self.mlp_Wc_) + self.mlp_bc_  # affine transformation
        hc = tf.tanh(hc)  # non-linearity
        hc = tf.matmul(hc, self.mlp_Wc) + self.mlp_bc  # affine transformation [B * M, Vy]

        c = tf.nn.sigmoid(hc)

        c = tf.reshape(c, [batch_size, longest_y, 1, 1])  #[BxNx1x1]

        ones = tf.ones(tf.shape(c))
        c_inverted = tf.subtract(ones, c)  #[BxNx1x1]

        # First we make the input to the MLP 2-D.
        # Every output row will be of size Vy, and after a softmax
        # will sum to 1.0.

        e_input = tf.reshape(x_embedded, [batch_size *  longest_x, self.emb_dim])

        # Here we apply the MLP to our input.
        he = tf.matmul(e_input, self.mlp_We_) + self.mlp_be_  # affine transformation
        he = tf.tanh(he)  # non-linearity
        he = tf.matmul(he, self.mlp_We) + self.mlp_be  # affine transformation [B * M, Vy]

        py_xa = tf.nn.softmax(he)
        py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size])  #[BxMxVy]

        # You could also use TF fully connected to create the MLP.
        # Then you don't have to specify all the weights and biases separately.
        # h = tf.contrib.layers.fully_connected(mlp_input, self.mlp_dim, activation_fn=tf.tanh, trainable=True)
        # h = tf.contrib.layers.fully_connected(h, self.y_vocabulary_size, activation_fn=None, trainable=True)

        # Now we perform a softmax which operates on a per-row basis.

        # First we make the input to the MLP 2-D for C.
        # Every output row will be of size Vy, and after a sigmoid
        # we will get non-discrete version of C.

        f_input = tf.reshape(y_embedded, [batch_size * longest_y, self.emb_dim])

        # Here we apply the MLP to our input.
        hf = tf.matmul(f_input, self.mlp_Wf_) + self.mlp_bf_  # affine transformation
        hf = tf.tanh(hf)  # non-linearity
        hf = tf.matmul(hf, self.mlp_Wf) + self.mlp_bf  # affine transformation [B * M, Vy]

        py_yprev = tf.nn.softmax(hf)
        py_yprev = tf.reshape(py_yprev, [batch_size, longest_y, self.y_vocabulary_size]) #[BxNxVy]


        # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)

        # Here comes a rather fancy matrix multiplication.
        # Note that tf.matmul is defined to do a matrix multiplication
        # [N, M] @ [M, Vy] for each item in the first dimension B.
        # So in the final result we have B matrices [N, Vy], i.e. [B, N, Vy].
        #
        # We matrix-multiply:
        #   pa_x       Shape: [B, N, *M*]
        # and
        #   py_xa      Shape: [B, *M*, Vy]
        # to get
        #   py_x  Shape: [B, N, Vy]
        #
        # Note: P(y|x) = prod_j p(y_j|x) = prod_j sum_aj p(aj|m)p(y_j|x_aj)
        #
        py_ax = tf.matmul(pa_x, py_xa)  # Shape: [B, N, Vy]
        py_ax = tf.expand_dims(py_ax, 2)  # [B, N, 1, Vy]

        term_c0 = tf.matmul(c, py_ax)
        term_c0 = tf.reshape(term_c0, [batch_size, longest_y, self.y_vocabulary_size])  # [B, N, Vy]

        py_yprev = tf.expand_dims(py_yprev, 2)  # [B, N, 1, Vy]
        term_c1 = tf.matmul(c_inverted, py_yprev)
        term_c1 = tf.reshape(term_c1, [batch_size, longest_y, self.y_vocabulary_size])  # [B, N, Vy]

        py_x = tf.add(term_c0, term_c1)
        ###############################################################################################################
        # Predictions, use c not p(c)

        # Round p(c) to get discrete classifier c.
        c = tf.round(c)
        c_inverted = tf.round(c_inverted)

        # Now calculate the terms for prediction with discrete classifier c
        term_c0 = tf.matmul(c, py_ax)
        term_c1 = tf.matmul(c_inverted, py_yprev)
        term_c0 = tf.reshape(term_c0, [batch_size, longest_y, self.y_vocabulary_size])
        term_c1 = tf.reshape(term_c1, [batch_size, longest_y, self.y_vocabulary_size])

        # This calculates the accuracy, i.e. how many predictions we got right.
        predictions = tf.argmax(term_c0, axis=2) + tf.argmax(term_c1, axis=2)  # [B, N]
        acc = tf.equal(predictions, self.y)
        acc = tf.cast(acc, tf.float32) * y_mask
        acc_correct = tf.reduce_sum(acc)
        acc_total = tf.reduce_sum(y_mask)
        acc = acc_correct / acc_total
        ###############################################################################################################
        # loss, use p(y|x), which is marginalized over alignment and collocation probabilities

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.y, [-1]),
            logits=tf.log(tf.reshape(py_x, [batch_size * longest_y, self.y_vocabulary_size])),
            name="logits"
        )
        cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
        cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
        cross_entropy = tf.reduce_mean(cross_entropy, axis=0)
        ###############################################################################################################
        # return this
        self.pa_x = pa_x
        self.py_x = py_x
        self.py_xa = py_xa
        self.py_yprev = py_yprev
        self.c = c
        self.loss = cross_entropy
        self.predictions = predictions
        self.accuracy = acc
        self.accuracy_correct = tf.cast(acc_correct, tf.int64)
        self.accuracy_total = tf.cast(acc_total, tf.int64)
        ###############################################################################################################

    def get_viterbi(self, x, y):
        """Returns the Viterbi alignment for (x, y)"""

        feed_dict = {
            self.x: x,  # English
            self.y: y  # French
        }

        # run model on this input
        py_xa, acc_correct, acc_total, loss, c = self.session.run(
            [self.py_xa, self.accuracy_correct, self.accuracy_total, self.loss, self.c],
            feed_dict=feed_dict)

        # things to return
        batch_size, longest_y = y.shape
        alignments = np.zeros((batch_size, longest_y), dtype="int64")
        probabilities = np.zeros((batch_size, longest_y), dtype="float32")

        for b, sentence in enumerate(y):
            for j, french_word in enumerate(sentence):
                if french_word == 0:  # Padding
                    break

                if c[b, j] == 1:
                    # when produced from source, get the alignment
                    probs = py_xa[b, :, y[b, j]]
                    a_j = probs.argmax()
                    p_j = probs[a_j]
                else:
                    # when produced from target, align to null
                    probs = py_xa[b, :, y[b, j]]
                    a_j = 0
                    p_j = probs[a_j]

                alignments[b, j] = a_j
                probabilities[b, j] = p_j

        return alignments, probabilities, acc_correct, acc_total, loss
