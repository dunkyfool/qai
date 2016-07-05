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

def cnn1d(x,w,b,para):
  s = [1,1,para['stride'],1]
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
  mean, var = tf.nn.moments(x,[0])
  return tf.nn.batch_normalization(x,mean,var,offset=beta,scale=gamma,variance_epsilon=1e-5)

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

def cnn_relu_bn(x,w,b,conv_para,gamma,beta):
  output1 = relu(cnn(x,w,b,conv_para))
  shape = output1.get_shape().as_list()
  bn = tf.reshape(output1,[-1,shape[-1]])
  output2 = batchnorm(bn,gamma,beta)
  output = tf.reshape(output2,[-1,shape[1],shape[2],shape[3]])
  return output

def cnn1d_relu_bn(x,w,b,conv_para,gamma,beta):
  output1 = relu(cnn1d(x,w,b,conv_para))
  shape = output1.get_shape().as_list()
  bn = tf.reshape(output1,[-1,shape[-1]])
  output2 = batchnorm(bn,gamma,beta)
  output = tf.reshape(output2,[-1,shape[1],shape[2],shape[3]])
  return output

def cnn_relu_maxpool_bn(x,w,b,conv_para,pool_para,gamma,beta):
  output1 = maxpool(relu(cnn(x,w,b,conv_para)),pool_para)
  shape = output1.get_shape().as_list()
  print shape,type(shape)
  bn = tf.reshape(output1,[-1,shape[-1]])
  output2 = batchnorm(bn,gamma,beta)
  output = tf.reshape(output2,[-1,shape[1],shape[2],shape[3]])
  return output

def dnn_relu(x,w,b):
  return relu(dnn(x,w,b))

def dnn_relu_dropout(x,w,b,p):
  return dropout(relu(dnn(x,w,b)),p)

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
