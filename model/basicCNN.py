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

class modelX():
  def __init__(self):
    self.x = tf.placeholder(tf.float32,[None,32,32,3])
    self.y_hat = tf.placeholder(tf.float32,[None, 10])
    self.f1 = weight([3,3,3,3])
    self.fb1 = bias([3])
    self.f2 = weight([3,3,3,3])
    self.fb2 = bias([3])
    self.f3 = weight([3,3,3,3])
    self.fb3 = bias([3])
    self.f4 = weight([3,3,3,3])
    self.fb4 = bias([3])
    self.w1 = weight([192,512])
    self.b1 = bias([512])
    self.w2 = weight([512,10])
    self.b2 = bias([10])
    self.conv_para = {'stride':1,
                     'pad':'SAME'}
    self.pool_para = {'stride':2,
                      'pad':'SAME',
                      'kernel':2}

    self.cnn1 = cnn_relu_maxpool(self.x,self.f1,self.fb1,self.conv_para,self.pool_para)
    #print self.cnn1.get_shape()
    self.cnn2 = cnn_relu_maxpool(self.cnn1,self.f2,self.fb2,self.conv_para,self.pool_para)
    self.cnn3 = cnn_relu(self.cnn2,self.f3,self.fb3,self.conv_para)
    self.cnn4 = cnn_relu(self.cnn3,self.f4,self.fb4,self.conv_para)
    self.cnn4_output = tf.reshape(self.cnn4,[-1,64*3])
    #self.cnn1_output = tf.reshape(self.cnn1,[-1,16*16*3])
    self.dnn1 = dnn_relu(self.cnn4_output,self.w1,self.b1)
    self.dnn2 = dnn(self.dnn1, self.w2, self.b2)
    self.softmax = softmax(self.dnn2)

  def loss(self,X,y,lr=2e-4):
    cross_entropy = -tf.reduce_sum(self.y_hat*tf.log(self.softmax))
    #f = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    f = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
    #f = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    for i in range(30):
      for j in range(10):
        batch_xs, batch_ys = random_minibatch(X,y,10)
        sess.run(f,feed_dict={self.x:batch_xs,self.y_hat:batch_ys})
        if j%10==0:
          pred = tf.equal(tf.argmax(self.softmax,1),tf.argmax(self.y_hat,1))
          acc = tf.reduce_mean(tf.cast(pred,tf.float32))
          print i,j,sess.run([cross_entropy, acc],feed_dict={self.x:X,self.y_hat:y})


if __name__=='__main__':
  pass
