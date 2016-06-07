import tensorflow as tf
import math

def weight(shape):
  w = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(w)

def bias(shape):
  b = tf.constant(0.1, shape=shape)
  return tf.Variable(b)

