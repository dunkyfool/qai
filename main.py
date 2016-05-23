import tensorflow as tf
import numpy as np

####################
# Random Minibatch #
####################
def random_minibatch(data,label,size):
  new_data = np.zeros((size,data.shape[1]))
  new_label = np.zeros((size,label.shape[1]))
  for i in range(size):
    idx = np.floor(np.random.uniform(low=0,high=data.shape[0])).astype(np.int64)
    new_data[i] += data[idx]
    new_label[i] += label[idx]
  return new_data, new_label


#############
# Load Data #
#############
train_data = np.asarray([0,0,1,0,0,1,1,1]).reshape((4,2))
train_label = np.asarray([1,0,0,1,0,1,1,0]).reshape((4,2))
#print 'train_data'; print train_data
#print 'train_lable'; print train_label

############
# Variable #
############
lr = 1e-1
epoch = 1000

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.truncated_normal([2, 2],stddev=1e-5))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############
# Training #
############
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(epoch):
  #batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs, batch_ys = random_minibatch(train_data,train_label,2)
  _, loss, acc = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
  if i%100 == 0:
    print loss, acc

