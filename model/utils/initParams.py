import tensorflow as tf

def weight(shape):
  w = tf.truncated_normal(shape, stddev=1e-3)
  return tf.Variable(w)

def bias(shape):
  b = tf.constant(0.0, shape=shape)
  return tf.Variable(b)

