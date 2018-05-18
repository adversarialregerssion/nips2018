import numpy as np
import tensorflow as tf

class Fcnn2(object):

    def __init__(self, params):
        self.params = params

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="weight")

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="bias")

    def fc_layer(self, input, insize, outsize, lname='1', act='relu'):
        with tf.name_scope(lname):
            sigma = np.sqrt(2. / (insize))
            W = tf.Variable(tf.truncated_normal(shape=[insize, outsize],
                                                      mean=0., stddev=sigma))
            b = tf.Variable(tf.zeros(outsize))
            conv1 = tf.matmul(input, W) + b
            if act == 'relu':
                fc_out = tf.nn.relu(conv1)
            elif act == 'sigmoid':
                fc_out = tf.nn.sigmoid(conv1)

        return fc_out

    def convlayer(self, input, patchsize, insize, outsize, strides=[1, 1, 1, 1], lname='1', act='relu'):
        with tf.name_scope(lname):
            sigma = np.sqrt(2. / (patchsize * patchsize * insize))
            conv1_w = tf.Variable(tf.truncated_normal(shape=[patchsize, patchsize, insize, outsize],
                                                      mean=0., stddev=sigma))
            conv1_b = tf.Variable(tf.zeros(outsize))
            conv1 = tf.nn.conv2d(input, conv1_w, strides=strides, padding='SAME') + conv1_b
            if act == 'relu':
                conv1 = tf.nn.relu(conv1)
            elif act == 'sigmoid':
                conv1 = tf.nn.sigmoid(conv1)

        return conv1

    def upsample(self, input, outsize, lname='1'):
        with tf.name_scope(lname):
            out = tf.image.resize_bicubic(input, size=outsize)
        return out

    def autoencoder(self, input):
        image_size = self.params["image_dims"]
        X = tf.reshape(input, [-1, image_size[0], image_size[1], 1])


        with tf.name_scope('encoder'):
            Xpreproc = tf.scalar_mul(2., tf.add(X, -0.5))  # dynamic range = [-1, 1]
            enc1 = self.convlayer(Xpreproc, 3, 1, 8, strides=[1, 2, 2, 1], lname='L1')  # (16 x 16)
            # enc2 = self.convlayer(enc1, 3, 32, 4, strides=[1, 1, 1, 1], lname='L2')
            out_enc = self.convlayer(enc1, 3, 8, 8, strides=[1, 2, 2, 1], lname='L3') # (4x4)

        with tf.name_scope('fully_connected'):
            N_bottleneck = 16

            fc1 = self.fc_layer(tf.contrib.layers.flatten(out_enc), 8*8*8, N_bottleneck, lname='1', act='relu')
            fc2 = self.fc_layer(fc1, N_bottleneck, N_bottleneck, lname='2', act='relu')

            reshaped_features = tf.reshape(fc2, [-1, 1, 1, N_bottleneck])
            out_fc = tf.tile(reshaped_features, [1, 8, 8, 1])

        with tf.name_scope('decoder'):
            # dec1 = self.convlayer(out_fc, 3, 8, 32, strides=[1, 1, 1, 1], lname='L1')
            dec1 = self.convlayer(out_fc, 3, N_bottleneck, 8, strides=[1, 1, 1, 1], lname='L1')  # (8 x 8)
            dec2 = self.upsample(dec1, outsize=[8, 8], lname='L1b')  # (24 x 24)
            dec3 = self.convlayer(dec2, 3, 8, 32, strides=[1, 1, 1, 1], lname='L2')  # (16 x 16)
            dec4 = self.upsample(dec3, outsize=[32, 32], lname='L2b')  # (48 x 48)
            out_dec = self.convlayer(dec4, 3, 32, 1, strides=[1, 1, 1, 1], lname='L3', act='sigmoid')  # (32 x 32)

        # out = tf.contrib.layers.flatten(out_dec)
        out = tf.identity(out_dec, name='output')

        with tf.name_scope('autoencoder') as scope:
            loss = tf.reduce_mean(tf.squared_difference(input, out))

        return loss, out, out_enc
