import numpy as np
import tensorflow as tf
from utils.layers import *
from utils.initParams import *
#
#  model1: CNN3x3 - maxpool - CNN3x3 - maxpool - CNN3x3 - CNN3x3 FC FC softmax
#  model2: CNN3x3 - maxpool - CNN3x3 - maxpool - CNN1x1 - CNN3x3 FC FC softmax
#  model3: CNN3x3 - maxpool - CNN3x3 - maxpool - CNN3x3 - CNN1x1 FC FC softmax
#

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

class model1():

  def __init__(self,input_dim=(32,32,3),num_class=10,kernel=[3,3],banks=[3,3],hidden=[100],batch=2):
    #variable constant
    H, W, C = input_dim
    k = kernel
    current_H, current_W = H, W
    self.bs = batch
    self.cnn_para = {}
    self.dnn_para = {}
    self.cnn_bn = {}
    self.dnn_bn = {}

    #get num of layer 
    banks = [C] + banks # add image channel as first filter banks
    self.num_cnn = len(banks) - 1
    self.cnn_shape = H/2**(self.num_cnn) * W/2**(self.num_cnn) * banks[self.num_cnn] # last cnn num of feature 
    hidden = [self.cnn_shape] + hidden + [num_class] # first neuron = last cnn; last neuron = num_class
    self.num_dnn = len(hidden) -1

    #variable cnn weight/bias
    # conv1: W1,b1, conv2: W2,b2 ...
    # batchnorm: W1: (N,H,W,C) b1: (C,)
    for i in range(self.num_cnn):
      current_H /= 2
      current_W /= 2
      self.cnn_para["W{0}".format(i+1)] = weight((k[i],k[i],banks[i],banks[i+1]))
      self.cnn_para["b{0}".format(i+1)] = bias([banks[i+1]])
      self.cnn_bn["W{0}".format(i+1)] = weight((self.bs,current_H,current_W,banks[i+1]))
      self.cnn_bn["b{0}".format(i+1)] = bias([banks[i+1]])
      self.cnn_bn["s{0}".format(i+1)] = (self.bs,current_H,current_W,banks[i+1])

    #variable dnn weight/bias
    # dnn1: W1, b1, dnn2: W1, b1 ...
    for i in range(self.num_dnn):
      self.dnn_para['W{0}'.format(i+1)] = weight((hidden[i],hidden[i+1]))
      self.dnn_para['b{0}'.format(i+1)] = bias([hidden[i+1]])
      self.dnn_bn['W{0}'.format(i+1)] = weight((self.bs,hidden[i+1]))
      self.dnn_bn['b{0}'.format(i+1)] = bias([hidden[i+1]])

    #variable para
    # feed same para in all cnn layer
    self.conv_para = {'stride':1,
                     'pad':'SAME'}
    self.pool_para = {'stride':2,
                      'pad':'SAME',
                      'kernel':2}

  def loss(self,X,y,mode,lr=1e-1,ep=2):
    _, H, W, C = X.shape
    _, cls = y.shape
    x = tf.placeholder(tf.float32, [self.bs,H,W,C])
    y_ = tf.placeholder(tf.float32, [self.bs,cls])
    self.cnn_input = {'0':x}
    self.dnn_input = {}
    lr = lr
    iters = ep * _ / self.bs

    #CNN layers
    for i in range(self.num_cnn):
       # cnn_relu_maxpool
      self.cnn_input['{0}'.format(i+1)] = cnn_relu_maxpool(self.cnn_input['{0}'.format(i)],
                                                           self.cnn_para['W{0}'.format(i+1)],
                                                           self.cnn_para['b{0}'.format(i+1)],
                                                           self.conv_para,
                                                           self.pool_para)
      # cnn_relu_maxpool_batchnorm
#      self.cnn_input['{0}'.format(i+1)] = cnn_relu_maxpool_bn(self.cnn_input['{0}'.format(i)],
#                                                           self.cnn_para['W{0}'.format(i+1)],
#                                                           self.cnn_para['b{0}'.format(i+1)],
#                                                           self.conv_para,
#                                                           self.pool_para,
#                                                           self.cnn_bn['W{0}'.format(i+1)],
#                                                           self.cnn_bn['b{0}'.format(i+1)],
#                                                           self.cnn_bn['s{0}'.format(i+1)])

    #CNN output feature
    cnn_output = self.cnn_input[str(self.num_cnn)]
    self.dnn_input['0'] = tf.reshape(cnn_output,[-1,self.cnn_shape])

    #DNN layers
    for i in range(self.num_dnn):
       # dnn_relu
      self.dnn_input['{0}'.format(i+1)] = dnn_relu(self.dnn_input['{0}'.format(i)],
                                                   self.dnn_para['W{0}'.format(i+1)],
                                                   self.dnn_para['b{0}'.format(i+1)])
      # dnn_relu_bn
#      self.dnn_input['{0}'.format(i+1)] = dnn_relu_bn(self.dnn_input['{0}'.format(i)],
#                                                   self.dnn_para['W{0}'.format(i+1)],
#                                                   self.dnn_para['b{0}'.format(i+1)],
#                                                   self.dnn_bn['W{0}'.format(i+1)],
#                                                   self.dnn_bn['b{0}'.format(i+1)])
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
      batch_xs, batch_ys = random_minibatch(X,y,self.bs)
      output = sess.run([score],feed_dict={x:batch_xs})[0]
      print output;print output.shape
    elif mode == 'train':
      print 'training start!!'
      for i in range(iters):
        batch_xs, batch_ys = random_minibatch(X,y,self.bs)
        _, loss, acc = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        if i%100 == 0:
            print loss; print acc*100

class modelX():
  def __init__(self):
    self.x = tf.placeholder(tf.float32,[None,28,28,1])
    self.y_hat = tf.placeholder(tf.float32,[None, 10])
    self.f1 = weight([3,3,1,1])
    self.fb = bias([1])
    self.w1 = weight([196,256])
    self.b1 = bias([256])
    self.w2 = weight([256,10])
    self.b2 = bias([10])
    self.conv_para = {'stride':1,
                     'pad':'SAME'}
    self.pool_para = {'stride':2,
                      'pad':'SAME',
                      'kernel':2}


    self.gamma = weight([2,14,14,1])
    self.beta = bias([1])


    self.cnn1 = cnn_relu_maxpool(self.x,self.f1,self.fb,self.conv_para,self.pool_para)
    shape = 2,14,14,1
    #self.cnn1 = cnn_relu_maxpool_bn(self.x,self.f1,self.fb,self.conv_para,self.pool_para,self.gamma,self.beta,shape)
    print self.cnn1.get_shape()
    self.cnn1_output = tf.reshape(self.cnn1,[-1,196])
    self.dnn1 = dnn_relu(self.cnn1_output,self.w1,self.b1)
    #self.dnn1 = dnn_relu_bn(self.cnn1_output,self.w1,self.b1,self.gamma,self.beta)
    self.dnn2 = dnn(self.dnn1, self.w2, self.b2)
    self.softmax = softmax(self.dnn2)

  def loss(self,X,y,lr=3e-3):
    cross_entropy = -tf.reduce_sum(self.y_hat*tf.log(self.softmax))
    #f = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    f = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
    #f = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    for i in range(10):
      for j in range(50):
        batch_xs, batch_ys = random_minibatch(X,y,2)
        sess.run(f,feed_dict={self.x:batch_xs,self.y_hat:batch_ys})
        #sess.run(f,feed_dict={self.x:X[j*2:(j+1)*2],
        #                      self.y_hat:y[j*2:(j+1)*2]})
        if j%10==0:
          pred = tf.equal(tf.argmax(self.softmax,1),tf.argmax(self.y_hat,1))
          acc = tf.reduce_mean(tf.cast(pred,tf.float32))
          loss, real_acc = 0,0
          for k in range(50):
            _loss, _acc = sess.run([cross_entropy, acc],feed_dict={self.x:X[k*2:(k+1)*2],self.y_hat:y[k*2:(k+1)*2]})
            loss += _loss
            real_acc +=_acc
          print('Loss: %.10f Acc: %.5f' %(loss,real_acc/50))

          #print sess.run([cross_entropy, acc],feed_dict={self.x:X,self.y_hat:y})


def test():
  x = np.random.random((100,32,32,3))
  y = np.random.random((100,10))

  _,H,W,C = x.shape
  _,cls = y.shape

  x = x-np.mean(x,axis=0)
  net = model1(input_dim=(H,W,C),num_class=cls,batch=1)
  net.loss(x,y,'train')
  pass

if __name__=='__main__':
  #test()
  pass
