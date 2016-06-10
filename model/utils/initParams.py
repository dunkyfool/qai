import tensorflow as tf
import numpy as np
import math

def weight(shape):
  if len(shape)==2:
    fin = shape[0]
    fout = shape[1]
  elif len(shape)==4:
    fin = shape[0]*shape[1]*shape[2]
    fout = shape[0]*shape[1]*shape[3]
  low = -4*np.sqrt(1.0/(fin+fout)) # use 4 for sigmoid, 1 for tanh activation 
  high = 4*np.sqrt(1.0/(fin+fout))
  return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
  #w = tf.truncated_normal(shape, stddev=std)
  #return tf.Variable(w)

def bias(shape):
  b = tf.constant(0.1, shape=shape)
  return tf.Variable(b)

