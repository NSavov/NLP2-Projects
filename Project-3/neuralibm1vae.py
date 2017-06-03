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

class NeuralIBM1ModelAER(NeuralIBM1Model):
    """Our Neural IBM1 model with latent collocation variable trough a variational autoencoder."""

    def __init__(self, batch_size=8,
                 x_vocabulary=None, y_vocabulary=None,
                 emb_dim=32, mlp_dim=64,
                 session=None):
        super().__init__(batch_size, x_vocabulary, y_vocabulary, emb_dim, mlp_dim, session)

    def _create_weights(self):
        """Create weights for the model."""
        with tf.variable_scope("MLP") as scope:
            # a(f_(j-1)) for Beta FFNN
            self.mlp_Waf_ = tf.get_variable(
                name="Waf_", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.mlp_dim])

            self.mlp_baf_ = tf.get_variable(
                name="baf_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Waf = tf.get_variable(
                name="Waf", initializer=glorot_uniform(),
                shape=[self.mlp_dim, 1])

            self.mlp_baf = tf.get_variable(
                name="baf", initializer=tf.constant_initializer(0.0),
                shape=[1])

            # b(f_(j-1)) for Beta FFNN
            self.mlp_Wbf_ = tf.get_variable(
                name="Wbf_", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.mlp_dim])

            self.mlp_bbf_ = tf.get_variable(
                name="bbf_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Wbf = tf.get_variable(
                name="Wbf", initializer=glorot_uniform(),
                shape=[self.mlp_dim, 1])

            self.mlp_bbf = tf.get_variable(
                name="bbf", initializer=tf.constant_initializer(0.0),
                shape=[1])

            # a(f_(j, j-1)) for Kuma FFNN
            self.mlp_Waff_ = tf.get_variable(
                name="Waff_", initializer=glorot_uniform(),
                shape=[self.emb_dim * 2, self.mlp_dim])

            self.mlp_baff_ = tf.get_variable(
                name="baff_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Waff = tf.get_variable(
                name="Waff", initializer=glorot_uniform(),
                shape=[self.mlp_dim, 1])

            self.mlp_baff = tf.get_variable(
                name="baff", initializer=tf.constant_initializer(0.0),
                shape=[1])

            # b(f_(j, j-1)) for Kuma FFNN
            self.mlp_Wbff_ = tf.get_variable(
                name="Wbff_", initializer=glorot_uniform(),
                shape=[self.emb_dim * 2, self.mlp_dim])

            self.mlp_bbff_ = tf.get_variable(
                name="bbff_", initializer=tf.constant_initializer(0.0),
                shape=[self.mlp_dim])

            self.mlp_Wbff = tf.get_variable(
                name="Wbff", initializer=glorot_uniform(),
                shape=[self.mlp_dim, 1])

            self.mlp_bbff = tf.get_variable(
                name="bbff", initializer=tf.constant_initializer(0.0),
                shape=[1])

    def _build_model(self):
        ###############################################################################################################
        # 0. Prelims

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

        ###############################################################################################################
        # 1. Let's create a (source, target, target-prev) word embeddings matrix.

        # create the (source) word embeddings matrix.
        # Shape: [Vx, emb_dim] where Vx is the source vocabulary size
        x_embeddings = tf.get_variable(
            name="x_embeddings", initializer=tf.random_uniform_initializer(),
            shape=[self.x_vocabulary_size, self.emb_dim])

        # create the (target, target-prev) word embeddings matrix.
        # shape: [Vy, emb_dim]
        y_embeddings = tf.get_variable(
            name="y_embeddings", initializer=tf.random_uniform_initializer(),
            shape=[self.y_vocabulary_size, self.emb_dim])

        # Now we start defining our graph.

        # This looks up the embedding vector for each word given the word IDs in self.x.
        # Shape: [B, M, emb_dim] where B is batch size, M is (longest) source sentence length.
        x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x)

        # This looks up the embedding vector for each word given the word IDs in self.x.
        # Shape: [B, N, emb_dim] where B is batch size, N is (longest) target sentence length.
        y_embedded = tf.nn.embedding_lookup(y_embeddings, self.y)

        # This looks up the embedding vector for each previous target word given the word IDs in self.y.
        # Shape: [B, N, emb_dim] where B is batch size, N is (longest) target sentence length.
        y_prev = tf.multiply(tf.ones([batch_size, 1], dtype="int64"), 2)
        y_padded = tf.concat([y_prev, self.y], 1)
        y_padded = y_padded[:, 0:-1]
        y_prev = tf.nn.embedding_lookup(y_embeddings, y_padded)
        ###############################################################################################################
        # 2a. The inference FFNN's for the (true) Beta posterior p(z|x,y) = Beta(a(y_(j-1)), b(y_(j-1)))
        #
        # For every French position this will predict the a and b parameter of the Beta distribution
        # This is used in the KL-divergence part of the ELBO for evaluation

        # MLP for a(f_(j-1))
        # reshape input to two dimensions
        mlp_input = tf.reshape(y_prev, [batch_size * longest_y, self.emb_dim])

        # Here we apply the MLP to our input.
        h = tf.matmul(mlp_input, self.mlp_Waf_) + self.mlp_baf_  # affine transformation [B * N, mlp_dim]
        h = tf.tanh(h)  # non-linearity
        h = tf.matmul(h, self.mlp_Waf) + self.mlp_baf  # affine transformation [B * N, 1]

        # Now we take the exponent of the result because a is always positive, and reshape back
        a_f = tf.exp(h)  # [B * N, 1]
        a_f = tf.reshape(a_f, [batch_size, longest_y, 1])

        # MLP for b(f_(j-1))
        # reshape input to two dimensions
        mlp_input = tf.reshape(y_prev, [batch_size * longest_y, self.emb_dim])

        # Here we apply the MLP to our input.
        h = tf.matmul(mlp_input, self.mlp_Wbf_) + self.mlp_bbf_  # affine transformation [B * N, mlp_dim]
        h = tf.tanh(h)  # non-linearity
        h = tf.matmul(h, self.mlp_Wbf) + self.mlp_bbf  # affine transformation [B * N, 1]

        # Now we take the exponent of the result because a is always positive, and reshape back
        b_f = tf.exp(h)  # [B * N, 1]
        b_f = tf.reshape(b_f, [batch_size, longest_y, 1])
        ###############################################################################################################
        # 2b. The inference FFNN's for the (approximate) Kuma posterior q(z|x,y) = Kuma(a(y_j, y_(j-1)), b(y_j, y_(j-1))
        #
        # For every French position given a French word and a previous French word this will predict a and b of Kuma
        # These parameters are used to define the inverse CDF of the Kuma distribution to draw a sample z ~ Kuma(a,b)
        # This sample will subsequently be used as the s parameter of a latent gatent neural IBM 1

        # concatenate french embeddings with the previous french embeddings
        y_cur_prev = tf.concat([y_embedded, y_prev], 2)

        # MLP for a(y_j, y_(j-1))
        # reshape input to two dimensions
        mlp_input = tf.reshape(y_cur_prev, [batch_size * longest_y, self.emb_dim * 2])

        # Here we apply the MLP to our input.
        h = tf.matmul(mlp_input, self.mlp_Waff_) + self.mlp_baff_  # affine transformation [B * N, mlp_dim]
        h = tf.tanh(h)  # non-linearity
        h = tf.matmul(h, self.mlp_Waff) + self.mlp_baff  # affine transformation [B * N, 1]

        # Now we take the exponent of the result because a is always positive, and reshape back
        a_ff = tf.exp(h)  # [B * N, 1]
        a_ff = tf.reshape(a_ff, [batch_size, longest_y, 1])

        # MLP for b(y_j, y_(j-1)
        # reshape input to two dimensions
        mlp_input = tf.reshape(y_cur_prev, [batch_size * longest_y, self.emb_dim * 2])

        # Here we apply the MLP to our input.
        h = tf.matmul(mlp_input, self.mlp_Wbff_) + self.mlp_bbff_  # affine transformation [B * N, mlp_dim]
        h = tf.tanh(h)  # non-linearity
        h = tf.matmul(h, self.mlp_Wbff) + self.mlp_bbff  # affine transformation [B * N, 1]

        # Now we take the exponent of the result because a is always positive, and reshape back
        b_ff = tf.exp(h)  # [B * N, 1]
        b_ff = tf.reshape(b_ff, [batch_size, longest_y, 1])
        ###############################################################################################################
        # sample from KUMA

