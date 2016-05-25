#
# ! self.cnn_shape => self.num_cnn should be num of max_pool
# ! all cnn layers' stride & pool size are fixed, add if else in loss cnn builder
#
import numpy as np
from layers import *
from tool.img2txt import loadFile

####################
# Random Minibatch #
####################
def random_minibatch(data,label,size):
  """
  Random choose minibatch from data and corresponding label
  * size : minibatch size
  """
  mask = [np.arange(size,data.shape[0])]
  new_data = np.zeros_like(data)
  new_data = np.delete(new_data,mask,axis=0)
  new_label = np.zeros_like(label)
  new_label = np.delete(new_label,mask,axis=0)

  for i in range(size):
    idx = np.floor(np.random.uniform(low=0,high=data.shape[0])).astype(np.int64)
    new_data[i] += data[idx]
    new_label[i] += label[idx]
  return new_data, new_label


def weight(shape):
  w = tf.truncated_normal(shape, stddev=1e-5)
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
  def __init__(self,input_dim=(48,48,3),num_class=10,kernel=[3,3,3,3],banks=[3,3,3,3],hidden=[100]):
    #variable constant
    H, W, C = input_dim
    k = kernel
    self.cnn_para = {}
    self.dnn_para = {}

    #get num of layer 
    banks = [C] + banks # add image channel as first filter banks
    self.num_cnn = len(banks) - 1
    self.cnn_shape = H/2**(self.num_cnn) * W/2**(self.num_cnn) * banks[self.num_cnn] # last cnn num of feature 
    hidden = [self.cnn_shape] + hidden + [num_class] # first neuron = last cnn; last neuron = num_class
    self.num_dnn = len(hidden) -1

    #variable cnn weight/bias
    # conv1: W1,b1, conv2: W2,b2 ...
    for i in range(self.num_cnn):
      self.cnn_para["W{0}".format(i+1)] = weight((k[i],k[i],banks[i],banks[i+1]))
      self.cnn_para["b{0}".format(i+1)] = bias([banks[i+1]])

    #variable dnn weight/bias
    # dnn1: W1, b1, dnn2: W1, b1 ...
    for i in range(self.num_dnn):
      self.dnn_para['W{0}'.format(i+1)] = weight((hidden[i],hidden[i+1]))
      self.dnn_para['b{0}'.format(i+1)] = bias([hidden[i+1]])

    #variable para
    # feed same para in all cnn layer
    self.conv_para = {'stride':1,
                     'pad':'SAME'}
    self.pool_para = {'stride':2,
                      'pad':'SAME',
                      'kernel':2}


  def loss(self,X,y,mode):
    _, H, W, C = X.shape
    _, cls = y.shape
    x = tf.placeholder(tf.float32, [None,H,W,C])
    y_ = tf.placeholder(tf.float32, [None,cls])
    self.cnn_input = {'0':x}
    self.dnn_input = {}
    lr = 4e-2
    epoch = 1000

    #CNN layers
    for i in range(self.num_cnn):
      self.cnn_input['{0}'.format(i+1)] = cnn_relu_maxpool(self.cnn_input['{0}'.format(i)],
                                                           self.cnn_para['W{0}'.format(i+1)],
                                                           self.cnn_para['b{0}'.format(i+1)],
                                                           self.conv_para,
                                                           self.pool_para)

    #CNN output feature
    cnn_output = self.cnn_input[str(self.num_cnn)]
    self.dnn_input['0'] = tf.reshape(cnn_output,[-1,self.cnn_shape])

    #DNN layers
    for i in range(self.num_dnn):
      self.dnn_input['{0}'.format(i+1)] = dnn_relu(self.dnn_input['{0}'.format(i)],
                                                   self.dnn_para['W{0}'.format(i+1)],
                                                   self.dnn_para['b{0}'.format(i+1)])
    #loss function
    score = softmax(self.dnn_input[str(self.num_dnn)])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(score), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #init
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    if mode == 'test':
      output = sess.run([score],feed_dict={x:X})[0]
      print output;print output.shape
    elif mode == 'train':
      print 'training start!!'
      for i in range(epoch):
        batch_xs, batch_ys = random_minibatch(X,y,2)
        _, loss, acc = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        if i%100 == 0:
            print loss, acc


def test():
  #x = np.random.random((5,48,48,3))
  #y = np.random.random((5,10))
  dic = loadFile('txt/')
  x, y = dic['data'], dic['label']
  _, H, W, C =  x.shape
  _, cls = y.shape
  net = CCnet(input_dim=(H,W,C),num_class=cls)
  net.loss(x,y,'train')
  pass

if __name__=='__main__':
  test()
  pass
