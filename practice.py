import numpy as np
import tensorflow as tf
from model.utils.layers import *

def test():
  batch_size,H,W,C = 200,50,60,3
  shape = [batch_size,H,W,C]

  x = tf.placeholder(tf.float32,shape)
  w = tf.Variable(tf.ones(shape))
  b = tf.Variable(tf.zeros([C]))
  y = tf.mul(x,b) + b

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  sess.run(init)

  _input = np.ones(shape)
  print sess.run([y],feed_dict={x:_input})
  pass

def check():
  N, D1, D2, D3 = 200, 50, 60, 3
  X = np.random.randn(N, D1)
  W1 = np.random.randn(D1, D2)
  W2 = np.random.randn(D2, D3)
  a = np.maximum(0, X.dot(W1)).dot(W2)

  print 'Before batch normalization:'
  print '  means: ', a.mean(axis=0)
  print '  stds: ', a.std(axis=0)
  print a.shape
  print 'After batch normalization (gamma=1, beta=0)'

  x = tf.placeholder(tf.float32,[N,D3])
  gamma = tf.Variable(tf.ones([N,D3]))
  beta = tf.Variable(11*tf.ones([D3]))
  batch_mean, batch_var = tf.nn.moments(x, [0])
  y = tf.nn.batch_normalization(x,batch_mean,batch_var,offset=beta,scale=gamma,variance_epsilon=1e-5)
  y_mean, y_var = tf.nn.moments(y,[0])
  #y_std = tf.std(y,axis=0)

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  sess.run(init)

  _mean, _var = sess.run([y_mean,y_var],feed_dict={x:a})
  print _mean
  print _var
  #a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
  #print '  mean: ', a_norm.mean(axis=0)
  #print '  std: ', a_norm.std(axis=0)
  pass


if __name__=='__main__':
  #test()
  check()
  pass
