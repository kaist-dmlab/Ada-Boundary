from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops


def norm(v):
    dim = len(v.get_shape())
    return tf.sqrt(tf.reduce_sum(v ** 2, axis=[i for i in range(dim - 1)], keep_dims=True))


def unit(v, eps=1e-8):
    vnorm = norm(v)
    return v / (vnorm + eps), vnorm


def xTy(x, y):
    dim = len(y.get_shape())
    xTy = tf.reduce_sum(x * y, axis=[i for i in range(dim - 1)], keep_dims=True, name="xTy")
    return xTy


def clip_by_norm(v, clip_norm):
    dim = len(v.get_shape())
    return tf.clip_by_norm(v, clip_norm, axes=[i for i in range(dim - 1)])


def gproj(y, g, normalize=False):
    # implementation of Eq.(6)
    if normalize:
        y, _ = unit(y)

    yTg = xTy(y, g)
    return g - (yTg * y)


def gexp(y, h, normalize=False):
    # implementation of Eq.(7)
    if normalize:
        y, _ = unit(y)
        h = gproj(y, h)

    u, hnorm = unit(h)
    return y * tf.cos(hnorm) + u * tf.sin(hnorm)


def gpt2(y, h1, h2, normalize=False):
    # implementation of Eq.(8)
    # parallel translation of tangent vector h1 toward h2
    if normalize:
        h1 = gproj(y, h1)
        h2 = gproj(y, h2)

    # svd(h2) = u * sigma * 1
    [u, unorm] = unit(h2)
    uTh1 = xTy(u, h1)
    return h1 - uTh1 * (tf.sin(unorm) * y + (1 - tf.cos(unorm)) * u)


def gpt(y, h, normalize=False):
    # implementation of Eq.(9)

    if normalize:
        h = gproj(y, h)

    [u, unorm] = unit(h)
    return (u * tf.cos(unorm) - y * tf.sin(unorm)) * unorm


class unit_initializer(init_ops.Initializer):
    def __init__(self, seed=None, dtype=tf.float32, eps=1e-8):
        self.seed = seed
        self.dtype = dtype
        self.eps = eps

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        v = random_ops.truncated_normal(shape, 0, 1.0, dtype, seed=self.seed)

        dim = len(v.get_shape())
        vnorm = tf.sqrt(tf.reduce_sum(v ** 2, axis=[i for i in range(dim - 1)], keep_dims=True))

        return v / (vnorm + self.eps)


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
