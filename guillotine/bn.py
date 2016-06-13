import numpy as np
import tensorflow as tf

def dnn(x,w,b):
  return tf.matmul(x, w) + b

def elementwise(x,w):
  return tf.mul(x,w)

def test():
  x = tf.placeholder(tf.float32,[None,10])
  w = tf.Variable(tf.ones([10,2]))

  w1 = tf.fill(tf.shape(x),0.1)
  bs = x.get_shape()[0]#tf.shape(x)[0]
  print bs
  shape = [bs,10]#tf.pack([bs,10])
  print shape
  w2 = tf.Variable(tf.ones(shape),validate_shape=False)

#  b = tf.Variable(tf.zeros([2]))
#  y = dnn(x,w,b)
  y = elementwise(x,w1)
  _input = -np.ones((3,10))

  init = tf.initialize_all_variables()
  sess = tf.InteractiveSession()
  sess.run(init)
  print sess.run([y],feed_dict={x:_input})
  print w1.get_shape(), type(w1), type(w), type(w2)
#  print w.get_shape()[0],type(w.get_shape()[0])
#  print w.get_shape().as_list()[0],type(w.get_shape().as_list()[0])
  pass

if __name__=='__main__':
  test()
  pass
