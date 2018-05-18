import numpy as np
import tensorflow as tf

class Conv(object):

    def __init__(self, params):
        self.params = params

    def autoencoder(self, input):

        # encoder layers:
        # The input to this layer is 32 x 32 x 3
        encoder_layer1 = tf.layers.conv2d(input, 8, [5, 5], strides=(2, 2), padding="SAME")
        # The output from this layer would be 16 x 16 x 8

        # The input to this layer is same as encoder_layer1 output: 16 x 16 x 8
        encoder_layer2 = tf.layers.conv2d(encoder_layer1, 16, [5, 5], strides=(2, 2), padding="SAME")
        # The output would be: 8 x 8 x 16

        # The input is same as above output: 8 x 8 x 16
        encoder_layer3 = tf.layers.conv2d(encoder_layer2, 32, [5, 5], strides=(4, 4), padding="SAME")
        # The output would be: 2 x 2 x 32
        # This is the latent representation of the input that is 128 dimensional.
        # Compression achieved from 32 x 32 x 3 i.e 3072 dimensions to 2 x 2 x 32 i. e. 128

        # decoder layers:
        # The input to this layer is 2 x 2 x 32
        decoder_layer1 = tf.layers.conv2d_transpose(encoder_layer3, 32, [5, 5], strides=(4, 4), padding="SAME")
        # Output from this layer: 8 x 8 x 32

        # The input to this layer: 8 x 8 x 32
        decoder_layer2 = tf.layers.conv2d_transpose(decoder_layer1, 16, [5, 5], strides=(2, 2), padding="SAME")
        # output from this layer: 16 x 16 x 16

        # The input of this layer: 16 x 16 x 16
        decoder_layer3 = tf.layers.conv2d_transpose(decoder_layer2, 3, [5, 5], strides=(2, 2), padding="SAME")
        # output of this layer: 32 x 32 x 3 # no. of channels are adjusted

        output = tf.identity(encoder_layer3, name = "encoded_representation") # the latent representation of the input image.


        y_pred = tf.identity(decoder_layer3, name = "output") # output of the decoder

        # define the loss for this model:
        # calculate the loss and optimize the network
        with tf.name_scope('autoencoder') as scope:
            loss = tf.reduce_mean(tf.squared_difference(input, y_pred))

        return loss, y_pred, None
