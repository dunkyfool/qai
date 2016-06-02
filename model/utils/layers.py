#
# build all combinational layers 
#
#
import tensorflow as tf
import numpy as np

################
# Normal Layer #
################
def dnn(x,w,b):
  return tf.matmul(x, w) + b

def cnn(x,w,b,para):
  s = [1,para['stride'],para['stride'],1]
  p = para['pad']
  return tf.nn.conv2d(x, w, strides=s, padding=p) + b

def rnn(x,w,b):
  pass

#################
# Special Layer #
#################
def dropout(x, p):
  return tf.nn.dropout(x,p)

def batchnorm(x, gamma, beta):
  g, b = gamma, beta
  return tf.nn.batch_normalization(x,0,1,offset=b,scale=g,variance_epsilon=1e-5)

def spatial_batchnorm(x, gamma, beta, shape):
  N, H, W, C = shape
  x = tf.reshape(x, [N*H*W,C])
  gamma = tf.reshape(gamma, [N*H*W,C])
  g, b = gamma, beta
  output = tf.nn.batch_normalization(x,0,1,offset=b,scale=g,variance_epsilon=1e-5)
  return tf.reshape(output, [N,H,W,C])

def maxpool(x, para):
  k = [1,para['kernel'],para['kernel'],1]
  s = [1,para['stride'],para['stride'],1]
  p = para['pad']
  return tf.nn.max_pool(x, k, strides=s, padding=p)

##############
# Activation #
##############
def relu(x):
  return tf.nn.relu(x)

def sigmoid(x):
  return tf.nn.sigmoid(x)

def tanh(x):
  return tf.nn.tanh(x)

def softmax(x):
  return tf.nn.softmax(x)

def log_softmax(x):
  return tf.nn.log_softmax(x)

#########
# Combo #
#########
def cnn_relu(x,w,b,conv_para):
  return relu(cnn(x,w,b,conv_para))

def cnn_relu_maxpool(x,w,b,conv_para,pool_para):
  return maxpool(relu(cnn(x,w,b,conv_para)),pool_para)

def cnn_relu_maxpool_bn(x,w,b,conv_para,pool_para,gamma,beta,shape):
  out = maxpool(relu(cnn(x,w,b,conv_para)),pool_para)
  return spatial_batchnorm(out, gamma, beta, shape)

def dnn_relu(x,w,b):
  return relu(dnn(x,w,b))

def dnn_relu_bn(x,w,b,gamma,beta):
  return batchnorm(relu(dnn(x,w,b)),gamma,beta)

########
# TEST #
########
def test():
  #variable
  data = np.ones((2,3,3,3))
  x = tf.placeholder(tf.float32, [None, 3,3,3])
  W = tf.Variable(tf.truncated_normal([2,3,3,3],stddev=1e-5))
  b = tf.Variable(tf.zeros([3]))

  #para
  conv_para = {'stride':1,
               'pad':'SAME'}
  pool_para = {'stride':1,
               'pad':'SAME',
               'kernel':2}
  bn_para = {'gamma':W,#same size as x
             'beta':b} #last dimension
  #y = dnn(x,W,b)
  #y = cnn(x,W,b,conv_para)
  #y = sigmoid(x)
  #y = log_softmax(x)
  #y = maxpool(x,pool_para)
  #y = dropout(x,0.5)
  #y = batchnorm(x,bn_para)

  #init
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  output = sess.run([y],feed_dict={x:data})[0]

  #test
  print output;print output.shape

########
# MAIN #
########
if __name__=='__main__':
  test()
  pass
