import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
FLAGS = tf.app.flags.FLAGS

#parameter for optimization
tf.app.flags.DEFINE_float('momentum', 0.9, '''momentum''')
tf.app.flags.DEFINE_float('learnRateDecay', 0.2, '''learnRateDecay''')
tf.app.flags.DEFINE_bool('nesterov', True, '''if True, use Nesterov momentum''')
tf.app.flags.DEFINE_bool('grassmann', True, '''if True, weights can be on G(1,n)''')
tf.app.flags.DEFINE_float('grad_clip', None, '''threshold of clipping gradient by norm''')
tf.app.flags.DEFINE_float('adam_beta2', 0.99, '''exponential decay rates for 2nd moment estimates in adamg''')
tf.app.flags.DEFINE_float('adam_eps', 1e-8, '''small constant to avoid dividing by 0 in adamg''')

# parameters for regularization
tf.app.flags.DEFINE_float('weightDecay', None,  '''L2 regularization for weights''')
tf.app.flags.DEFINE_float('biasDecay', None, '''L2 regularization for biases''')
tf.app.flags.DEFINE_float('gammaDecay', None,  '''L2 regularization for scaling parameters in BN''')
tf.app.flags.DEFINE_float('betaDecay', None, '''L2 regularization for offset parameters in BN''')
tf.app.flags.DEFINE_float('omega', None, '''orthogonality regularization for weight matrices''')

