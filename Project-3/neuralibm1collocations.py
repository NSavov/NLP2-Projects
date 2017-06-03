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

        y_embedded = tf.nn.embedding_lookup(y_embeddings, self.y)


        # 2. Now we define the generative model P(Y | X=x)

        # first we need to know some sizes from the current input data
        batch_size = tf.shape(self.x)[0]
        longest_x = tf.shape(self.x)[1]  # longest M
        longest_y = tf.shape(self.y)[1]  # longest N

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

        c = tf.reshape(c, [batch_size, longest_y, 1])
        c_expanded = tf.expand_dims(c, 2) #[BxNx1x1]
        c_e = tf.tile(c_expanded, [1, 1, longest_x,  self.y_vocabulary_size])  # [BxNxMxVy]

        c_inverted = tf.ones(tf.shape(c))
        c_f = tf.tile(c_inverted, [1, 1, self.y_vocabulary_size])  # [BxNxVy]

        # First we make the input to the MLP 2-D.
        # Every output row will be of size Vy, and after a softmax
        # will sum to 1.0.


        e_input = tf.reshape(x_embedded, [batch_size *  longest_x, self.emb_dim])

        # Here we apply the MLP to our input.
        he = tf.matmul(e_input, self.mlp_We_) + self.mlp_be_  # affine transformation
        he = tf.tanh(he)  # non-linearity
        he = tf.matmul(he, self.mlp_We) + self.mlp_be  # affine transformation [B * M, Vy]

        py_xa = tf.nn.softmax(he)
        py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size]) #[BxMxVy]

        py_xa_ext = tf.expand_dims(py_xa, 1) #[Bx1xMx1]
        py_xa_ext = tf.tile(py_xa_ext, [1, longest_y, 1,1])  # [BxNxMxVy]

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


        term_c0 = tf.multiply(c_e, py_xa_ext)  # [BxNxMxVy]
        term_c1 = tf.multiply(c_f, py_yprev)  # [BxNxVy]

        term_c1 = tf.expand_dims(term_c1, 2)
        term_c1 = tf.tile(term_c1, [1, 1, longest_x, 1]) # [BxNxMxVy]
        trans_prob = tf.add(term_c0, term_c1) # [BxNxMxVy]

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
        pa_x = tf.expand_dims(pa_x, 2)  # Shape: [B, N, 1, M]
        py_x = tf.matmul(pa_x, trans_prob)  # Shape: [B, N, 1, Vy]
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
