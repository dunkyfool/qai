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

def check_forward():
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

  z = batchnorm(x,gamma,beta)
  z_mean, z_var = tf.nn.moments(z,[0])

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  sess.run(init)

  print 'tf'
  _mean, _var = sess.run([y_mean,y_var],feed_dict={x:a})
  print _mean
  print _var

  print 'layer'
  _mean, _var = sess.run([z_mean,z_var],feed_dict={x:a})
  print _mean
  print _var
  pass

def check_backward():
  N, D = 4, 5
  x = 5 * np.random.randn(N, D) + 12
  gamma = np.random.randn(D)
  beta = np.random.randn(D)
  gamma_add = gamma[0] + 1e-5
  gamma_minus = gamma[0] - 1e-5
  #dout = np.random.randn(N, D)
  Y = np.random.randn(D)

  # variable
  data = tf.placeholder(tf.float32,[N,D],name='data')
  label = tf.placeholder(tf.float32,[D],name='label')
  g = tf.Variable(gamma,dtype=tf.float32,name='g')
  b = tf.Variable(beta,dtype=tf.float32,name='b')
  g_add_dx = tf.Variable(gamma_add,dtype=tf.float32,name='g_add')
  g_minus_dx = tf.Variable(gamma_minus,dtype=tf.float32,name='g_minus')

  # formula
  y = batchnorm(data,g,b)
  y_add = batchnorm(data,g_add_dx,b)
  y_minus = batchnorm(data,g_minus_dx,b)
  _z = tf.div(tf.sub(y_add,y_minus),2*1e-5)
  z = tf.reduce_sum(_z,0)
  loss = tf.square(tf.sub(y,label))

  grad = tf.gradients(loss,[g])[0]

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  sess.run(init)

  print sess.run([y],feed_dict={data:x})
  print sess.run([y_add],feed_dict={data:x})
  print sess.run([y_minus],feed_dict={data:x})
  print sess.run([_z],feed_dict={data:x})
  print sess.run([z],feed_dict={data:x})
  print sess.run([grad],feed_dict={data:x,label:Y})
  #for v in tf.all_variables():
  #  print v.value()
  pass

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
   """
   sample a few random elements and only return numerical
   in this dimensions.
   """

   for i in xrange(num_checks):
     ix = tuple([randrange(m) for m in x.shape])
     oldval = x[ix]
     x[ix] = oldval + h # increment by h
     fxph = f(x) # evaluate f(x + h)
     x[ix] = oldval - h # increment by h
     fxmh = f(x) # evaluate f(x - h)
     x[ix] = oldval # reset

     grad_numerical = (fxph - fxmh) / (2 * h)
     grad_analytic = analytic_grad[ix]
     rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
     print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)

if __name__=='__main__':
  #test()
  #check_forward()
  #check_backward()
  pass
