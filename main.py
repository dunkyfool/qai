import numpy as np
from layers import *

def weight(shape):
  w = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(w)

def bias(shape):
  b = tf.constant(0.0, shape=shape)
  return tf.Variable(b)

class CCnet(object):
  """
  Input (N,224,224,3)
  cnnv1_relu_maxpool
  conv2_relu_maxpool
  conv3_relu_maxpool
  conv4_relu_maxpool (down to 7x7 grid_cell)
  dnn1
  dnn2
  log_softmax
  """
  def __init__(self,input_dim=(224,224,3),output_dim=10,kernel=3,banks=[5,10,15,20]):
    #variable constant
    self.input_dim = input_dim
    H, W, C = input_dim
    cls = output_dim

    #variable weight/bias
    self.conv1_W = weight((kernel,kernel,C,banks[0]))
    self.conv1_b = bias([banks[0]])
    self.conv1_shape = H/2 * W/2 * banks[0]
#    conv2_W = weight()
#    conv2_b = bias()
#    conv3_W = weight()
#    conv3_b = bias()
#    conv4_W = weight()
#    conv4_b = bias()
    self.dnn1_W = weight((self.conv1_shape,cls))
    self.dnn1_b = bias((cls,))
#    dnn2_W = weight()
#    dnn2_b = bias()

    #variable para
    self.conv1_conv_para = {'stride':1,
                            'pad':'SAME'}
    self.conv1_pool_para = {'stride':2,
                            'pad':'SAME',
                            'kernel':2}

  def loss(self,X):
    _, H, W, C = X.shape
    x = tf.placeholder(tf.float32, [None,H,W,C])
    cnn1 = cnn_relu_maxpool(x,self.conv1_W,self.conv1_b,self.conv1_conv_para,self.conv1_pool_para)
    cnn1 = tf.reshape(cnn1,[-1,self.conv1_shape])
    dnn1 = dnn_relu(cnn1,self.dnn1_W,self.dnn1_b)
    y = log_softmax(dnn1)

    #init
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    output = sess.run([y],feed_dict={x:X})[0]
    print output;print output.shape


def test():
  x = np.ones((2,224,224,3))
  net = CCnet()
  net.loss(x)
  pass

if __name__=='__main__':
  test()
  pass
