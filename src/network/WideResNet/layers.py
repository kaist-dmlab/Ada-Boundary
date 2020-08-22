from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Variable Function
def _variable(name, shape, initializer, trainable):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def base_layer(input_tensor, shape, F, bias, layer_name):
  with tf.variable_scope(layer_name) as scope:
    weight = _variable('weight',
                 shape=shape,
                 initializer=tf.initializers.glorot_uniform(),
                 trainable=True)

    if bias:
      b = _variable('bias',
              shape=shape[-1],
              initializer=tf.zeros_initializer(),
              trainable=True)
      preactivation = F(input_tensor, weight) + b
    else:
      preactivation = F(input_tensor, weight)

    return preactivation

# Convolution Layer
def convolution_layer(input_tensor, shape, strides, bias, layer_name):
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME', use_cudnn_on_gpu=True)

  return base_layer(input_tensor, shape, conv2d, bias, layer_name)

# Fully Connected Layer
def full_connection_layer(input_tensor, shape, bias, layer_name):
  return base_layer(input_tensor, shape, tf.matmul, bias, layer_name)

# BatchNormalization Layer

def batch_norm(inputs, bn_param, scale=True, momentum=0.99, epsilon=1e-5, name='batch_norm'):
  with tf.variable_scope(name):
    beta = _variable('beta', [inputs.get_shape()[-1]],
               initializer=tf.zeros_initializer(),
               trainable=True)

    if scale:
      gamma = _variable('gamma', [inputs.get_shape()[-1]],
                  initializer=tf.ones_initializer(),
                  trainable=True)
    else:
      gamma = None

    reduced_dim = [i for i in range(len(inputs.get_shape())-1)]
    batch_mean, batch_var = tf.nn.moments(inputs,reduced_dim,keep_dims=False)

    # moving average of the populations
    pop_mean = _variable('pop_mean',
                   shape=[inputs.get_shape()[-1]],
                   initializer=tf.zeros_initializer(),
                   trainable=False)
    pop_var = _variable('pop_var',
                  shape=[inputs.get_shape()[-1]],
                  initializer=tf.ones_initializer(),
                  trainable=False)

    pop_mean_op = tf.assign(pop_mean, pop_mean * momentum + batch_mean * (1 - momentum))
    pop_var_op  = tf.assign(pop_var, pop_var * momentum + batch_var * (1 - momentum))

    tf.add_to_collection('batch_norm_update', pop_mean_op)
    tf.add_to_collection('batch_norm_update', pop_var_op)

    # for training, bn_param[0]=0
    # for evaluation, bn_param[0]=1
    mean = bn_param[0]*pop_mean + (1-bn_param[0])*batch_mean
    var = bn_param[0]*pop_var + (1-bn_param[0])*batch_var

    return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

# Useful Aggregated Layers
def bn_relu_conv(input_layer, filter_shape, strides, bn_param):
    bn = batch_norm(input_layer, bn_param)
    relu = tf.nn.relu(bn)
    conv = convolution_layer(relu, shape=filter_shape, strides = strides, bias=False, layer_name='conv')
    return conv, relu

def bn_relu_dropout_conv(input_layer, filter_shape, strides, bn_param, keep_prob):
    layer = [batch_norm(input_layer, bn_param)]
    layer.append(tf.nn.relu(layer[-1]))

    layer.append(tf.nn.dropout(layer[-1], keep_prob[0]))

    layer.append(convolution_layer(layer[-1], shape=filter_shape, strides=strides, bias=False, layer_name='conv'))
    return layer[-1]
