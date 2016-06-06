import tensorflow as tf
import math

def weight(shape):
  w = tf.truncated_normal(shape, stddev=1/math.sqrt(float(shape[0])))
  return tf.Variable(w)

def bias(shape):
  b = tf.constant(0.0, shape=shape)
  return tf.Variable(b)

