import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from network.WideResNet.layers import *
from network.WideResNet.grassmann_optimizer import *
from network.WideResNet.hyperparameters import FLAGS
from network.WideResNet.gutils import *

class WideResNet(object):
    def __init__(self, num_units, wide_factor, image_shape, num_labels, train_batch_size, test_batch_size):
        assert (num_units - 4) % 6 == 0, print('depth should be 6n+4')

        # Hyperparameter for wide-resnet
        self.n = int((num_units - 4) / 6)
        self.k = wide_factor

        # Variables for Input Data
        self.image_shape = image_shape
        self.num_labels = num_labels
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Input Data Meta Info
        [height, width, channels] = image_shape

        # Placeholder variables for training and testing
        train_batch_shape = [train_batch_size, height, width, channels]
        self.train_image_placeholder = tf.placeholder(
            tf.float32,
            shape=train_batch_shape,
            name='train_images'
        )
        self.train_label_placeholder = tf.placeholder(
            tf.int32,
            shape=[train_batch_size, ],
            name='train_labels'
        )
        self.train_weight_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, ],
            name='train_weight'
        )
        test_batch_shape = [test_batch_size, height, width, channels]
        self.test_image_placeholder = tf.placeholder(
            tf.float32,
            shape=test_batch_shape,
            name='test_images'
        )
        self.test_label_placeholder = tf.placeholder(
            tf.int32,
            shape=[test_batch_size, ],
            name='test_labels'
        )

    def first_residual_block(self, input_layer, output_channel, bn_param, keep_prob, down_sample=False):
        input_channel = input_layer.get_shape().as_list()[-1]
        assert input_channel != output_channel

        if down_sample:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, 1, 1, 1]

        with tf.variable_scope('layer1_in_block'):
            conv1, relu = bn_relu_conv(input_layer, [3, 3, input_channel, output_channel], strides=strides,
                                       bn_param=bn_param)
        with tf.variable_scope('layer2_in_block'):
            conv2 = bn_relu_dropout_conv(conv1, [3, 3, output_channel, output_channel], strides=[1, 1, 1, 1],
                                         bn_param=bn_param, keep_prob=keep_prob)

        projection = convolution_layer(relu, shape=[1, 1, input_channel, output_channel], strides=strides, bias=False,
                                       layer_name='projection')

        return conv2 + projection

    def residual_block(self, input_layer, output_channel, bn_param, keep_prob):
        input_channel = input_layer.get_shape().as_list()[-1]
        assert input_channel == output_channel

        with tf.variable_scope('layer1_in_block'):
            conv1, _ = bn_relu_conv(input_layer, [3, 3, input_channel, output_channel], strides=[1, 1, 1, 1],
                                    bn_param=bn_param)
        with tf.variable_scope('layer2_in_block'):
            conv2 = bn_relu_dropout_conv(conv1, [3, 3, output_channel, output_channel], strides=[1, 1, 1, 1],
                                         bn_param=bn_param, keep_prob=keep_prob)

        output = conv2 + input_layer

        return output

    def build_network(self, images, is_training, reuse):

        keep_prob = []
        bn_param = []
        if is_training:
            keep_prob.append(0.7)
            bn_param.append(0)
        else:
            keep_prob.append(1.0)
            bn_param.append(1)


        layers = []
        with tf.variable_scope('wide-resnet', reuse=reuse):
            with tf.variable_scope('group1'):
                conv0 = convolution_layer(images, shape=[3,3,self.image_shape[2],16], strides=[1,1,1,1], bias=False, layer_name='conv0')
                layers.append(conv0)

            for i in range(self.n):
                with tf.variable_scope('group2_block%d' %i):
                    if i == 0  and self.k != 1:
                        conv1 = self.first_residual_block(layers[-1], 16*self.k, bn_param, keep_prob, down_sample=False)
                    else:
                        conv1 = self.residual_block(layers[-1], 16*self.k, bn_param, keep_prob)
                    layers.append(conv1)

            for i in range(self.n):
                with tf.variable_scope('group3_block%d' % i):
                    if i == 0:
                        conv2 = self.first_residual_block(layers[-1], 32 * self.k, bn_param, keep_prob, down_sample=True)
                    else:
                        conv2 = self.residual_block(layers[-1], 32 * self.k, bn_param, keep_prob)
                    layers.append(conv2)

            for i in range(self.n):
                with tf.variable_scope('group4_block%d' % i):
                    if i == 0:
                        conv3 = self.first_residual_block(layers[-1], 64 * self.k, bn_param, keep_prob, down_sample=True)
                    else:
                        conv3 = self.residual_block(layers[-1], 64 * self.k, bn_param, keep_prob)
                    layers.append(conv3)
                #assert conv3.get_shape().as_list()[1:] == [8, 8, 64 * self.k]

            with tf.variable_scope('fc'):
                bn_layer = batch_norm(layers[-1], bn_param)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                assert global_pool.get_shape().as_list()[-1:] == [64 * self.k]

                shape = [global_pool.get_shape().as_list()[-1], self.num_labels]
                output = full_connection_layer(global_pool, shape=shape, bias=True, layer_name='output')

                layers.append(output)

            # output = [softmax matrix, logits]
            return tf.nn.softmax(layers[-1]), layers[-1]


    def build_train_op(self, lr_boundaries, lr_values, optimizer_type):
        train_step = tf.Variable(initial_value=0, trainable=False)
        self.train_step = train_step

        prob, logits = self.build_network(self.train_image_placeholder, True, False)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_label_placeholder,
            logits=logits
        )

        weighted_loss = tf.multiply(cross_entropy, self.train_weight_placeholder)
        cross_entropy_mean = tf.reduce_mean(weighted_loss, name='cross_entropy')

        # Accuracy Calculation
        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.train_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)

        ########################
        # variance -> distance
        mean, variance = tf.nn.moments(prob, axes=[1])

        # distance = sign(prediction) * variance
        # sign function : y = 2*prediction - 1
        sign = tf.subtract(tf.scalar_mul(2.0, prediction), 1.0)
        distance = sign * tf.sqrt(variance)
        ########################

        self.train_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        self.learning_rate = tf.train.piecewise_constant(train_step, lr_boundaries, lr_values)

        # Optimizer Setting
        if optimizer_type == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif optimizer_type == 'momentum':
            opt = tf.train.MomentumOptimizer(self.learning_rate, FLAGS.momentum, use_nesterov=FLAGS.nesterov)

        weight = [i for i in tf.trainable_variables() if 'weight' in i.name]
        bias = [i for i in tf.trainable_variables() if 'bias' in i.name]
        beta = [i for i in tf.trainable_variables() if 'beta' in i.name]
        gamma = [i for i in tf.trainable_variables() if 'gamma' in i.name]

        assert len(weight) + len(bias) + len(beta) + len(gamma) == len(tf.trainable_variables())

        grads, total_loss, cross_entropy_loss = self.train_graph_model(opt, cross_entropy_mean)
        train_op = self.build_graph_train(opt, grads, optimizer_type, train_step)

        return cross_entropy_loss, self.train_accuracy, train_op, cross_entropy, prob, distance


    def build_test_op(self):
        prob, logits = self.build_network(self.test_image_placeholder, False, True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.test_label_placeholder,
            logits=logits
        )
        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.test_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)
        self.test_loss = tf.reduce_mean(loss)
        self.test_accuracy = tf.reduce_mean(prediction)

        return  self.test_loss, self.test_accuracy, loss

    def train_graph_model(self, opt, cross_entropy_mean):
        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_losses = []
        tower_cross_entropy = []

        weight = [i for i in tf.trainable_variables() if 'weight' in i.name]
        bias = [i for i in tf.trainable_variables() if 'bias' in i.name]
        beta = [i for i in tf.trainable_variables() if 'beta' in i.name]
        gamma = [i for i in tf.trainable_variables() if 'gamma' in i.name]

        assert len(weight) + len(bias) + len(beta) + len(gamma) == len(tf.trainable_variables())

        if FLAGS.grassmann:
            for var in weight:
                undercomplete = np.prod(var.shape[0:-1]) > var.shape[-1]
                if undercomplete and ('conv' in var.name):
                    ## initialize to scale 1
                    var._initializer_op = tf.assign(var, unit_initializer()(var.shape)).op
                    tf.add_to_collection('grassmann', var)

        ## build graphs for regularization
        if FLAGS.omega is not None:
            for var in tf.get_collection('grassmann'):
                shape = var.get_shape().as_list()
                v = tf.reshape(var, [-1, shape[-1]])
                v_sim = tf.matmul(tf.transpose(v), v)

                eye = tf.eye(shape[-1])
                assert eye.get_shape() == v_sim.get_shape()

                orthogonality = tf.multiply(tf.reduce_sum((v_sim - eye) ** 2), 0.5 * FLAGS.omega,
                                                            name='orthogonality')
                tf.add_to_collection('orthogonality', orthogonality)

        if FLAGS.weightDecay is not None:
            for var in [i for i in weight if not i in tf.get_collection('grassmann')]:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.weightDecay, name='weightcost')
                tf.add_to_collection('weightcost', weight_decay)

        if FLAGS.biasDecay is not None:
            for var in bias:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.biasDecay, name='weightcost')
                tf.add_to_collection('weightcost', weight_decay)

        if FLAGS.gammaDecay is not None:
            for var in gamma:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.gammaDecay, name='weightcost')
                tf.add_to_collection('weightcost', weight_decay)

        if FLAGS.betaDecay is not None:
            for var in beta:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.betaDecay, name='weightcost')
                tf.add_to_collection('weightcost', weight_decay)

        if tf.get_collection('weightcost'):#, scope):
            weightcost = tf.add_n(tf.get_collection('weightcost', scope), name='weightcost')
        else:
            weightcost = tf.zeros([1])

        if tf.get_collection('orthogonality'):#, scope):
            orthogonality = tf.add_n(tf.get_collection('orthogonality', scope), name='orthogonality')
        else:
            orthogonality = tf.zeros([1])
            
        # Calculate the total loss for the current tower.
        total_loss = cross_entropy_mean + weightcost + orthogonality

        if opt != None:
        # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(total_loss)

            tower_grads.append(grads)
            tower_losses.append(total_loss)
            tower_cross_entropy.append(cross_entropy_mean)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        average_grads = average_gradients(tower_grads)

        losses = tf.reduce_mean(tower_losses)
        cross_entropy = tf.reduce_mean(tower_cross_entropy, name='cross_entropy')

        return average_grads, losses, cross_entropy

    def build_graph_train(self, opt, grads, optimizer_type, train_step):
        if optimizer_type == 'sgdg' or optimizer_type == 'adamg':
            grads_a = [i for i in grads if not i[1] in tf.get_collection('grassmann')]
            grads_b = [i for i in grads if i[1] in tf.get_collection('grassmann')]

            apply_gradient_op = opt.apply_gradients(grads_a, grads_b, global_step=train_step)
        else:
            apply_gradient_op = opt.apply_gradients(grads, global_step=train_step)

        return tf.group(*([apply_gradient_op] + tf.get_collection('batch_norm_update')))
