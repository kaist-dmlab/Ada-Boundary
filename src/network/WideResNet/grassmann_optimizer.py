from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from network.WideResNet.gutils import *


class HybridOptimizer(optimizer.Optimizer):
    def __init__(self, optimizer_a, optimizer_b, use_locking=False, name="HybridOptimizer"):
        super(HybridOptimizer, self).__init__(use_locking, name)
        self._optimizer_a = optimizer_a
        self._optimizer_b = optimizer_b

    def apply_gradients(self, grads_and_vars_a, grads_and_vars_b, global_step=None, name=None):
        op_a = self._optimizer_a.apply_gradients(grads_and_vars_a, name)
        op_b = self._optimizer_b.apply_gradients(grads_and_vars_b, name)
        return tf.group(*[op_a, op_b, global_step.assign_add(1)])


class SgdgOptimizer(optimizer.Optimizer):
    """Optimizer that implements stochastic gradient descent with momentum on G(1,n).

       References:
          - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
            (https://arxiv.org/abs/1709.09603)
    """

    def __init__(self, learning_rate, momentum, grad_clip=None, use_locking=False, name="SgdgOptimizer"):
        super(SgdgOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._grad_clip = grad_clip

        # Tensor versions of the constructor arguments, created in _prepare().
        self._learning_rate_t = None
        self._momentum_t = None
        self._grad_clip_t = None

    def _prepare(self):
        self._learning_rate_t = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._momentum_t = tf.convert_to_tensor(self._momentum, name="momentum")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t = None

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)

    def _apply_dense(self, grad, var):
        mom = self.get_slot(var, "momentum")

        unity, _ = unit(var)  # for numerical stability
        h = gproj(unity, grad)

        if self._grad_clip_t != None:
            h_hat = clip_by_norm(h, self._grad_clip_t)
        else:
            h_hat = h

        mom_new = self._momentum_t * mom - self._learning_rate_t * h_hat

        var_update = tf.assign(var, gexp(unity, mom_new))
        mom_update = tf.assign(mom, gpt(unity, mom_new))

        return tf.group(*[var_update, mom_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError()


def reduce_shape(p):
    dim = len(p.get_shape())
    return [1] * (dim - 1) + [p.get_shape().as_list()[-1]]


class AdamgOptimizer(optimizer.Optimizer):
    """Optimizer that implements Adam on G(1,n).

       References:
          - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
            (https://arxiv.org/abs/1709.09603)
    """

    def __init__(self, learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8, grad_clip=None, use_locking=False,
                 name="Adamg"):
        super(AdamgOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._grad_clip = grad_clip

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._grad_clip_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
        self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            dtype = v.dtype
            self._zeros_slot(v, "m", self._name)
            self._get_or_make_slot_with_initializer(v, tf.zeros_initializer(dtype), tf.TensorShape(reduce_shape(v)),
                                                    dtype, "v", self._name)

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = tf.convert_to_tensor(self._epsilon, name="epsilon")
        if self._grad_clip != None:
            self._grad_clip_t = tf.convert_to_tensor(self._grad_clip, name="grad_clip")
        else:
            self._grad_clip_t = None

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        unity, _ = unit(var)  # for numerical stability
        h = gproj(unity, grad)

        if self._grad_clip_t != None:
            h_hat = clip_by_norm(h, self._grad_clip_t)
        else:
            h_hat = h

        mnew = self._beta1_t * m + (1.0 - self._beta1_t) * h_hat
        vnew = self._beta2_t * v + (1.0 - self._beta2_t) * xTy(h_hat, h_hat)

        alpha = tf.sqrt(1 - self._beta2_power) / (1. - self._beta1_power)
        deltas = (-alpha * self._lr_t) * mnew / tf.sqrt(vnew + self._epsilon_t)

        var_update = tf.assign(var, gexp(unity, deltas))
        m_update = tf.assign(m, gpt2(unity, mnew, deltas))
        v_update = tf.assign(v, vnew)

        return tf.group(*[var_update, m_update, v_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError()

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with tf.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return tf.group(*update_ops + [update_beta1, update_beta2],
                        name=name_scope)