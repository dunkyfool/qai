import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class CNN_Layers(object):
    def __init__(self, image_size, batch_size, RGB_layers, filter_height, filter_width, in_channel, out_channel):
        self.input = input_data
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32,[None,image_size,image_size,RGB_layers])
        self.w = tf.Variable(tf.random_normal([filter_height, filter_width, in_channel, out_channel]))
        self.b = tf.Variable(tf.zeros([out_channel]))
        conv = tf.nn.conv2d(self.x,self.w,strides=[1,1,1,1],padding='SAME')
        act_conv = tf.nn.relu(conv)
        self.pooling_conv = tf.nn.max_pool(act_conv,ksize=[1,4,4,1],strides=[1,2,2,1],padding='SAME')
        self.pooling_conv_size_flatten = pow(image_size/2,2)*out_channel
        self.poolin_conv_size = image_size/2
class DNN_Layers(object):
    def __init__(self, input, in_neuron, out_neuron):
        flatten_input = tf.reshape(input,[-1, in_neuron])
        self.w = tf.Variable(tf.random_normal([in_neuron,out_neuron]))
        self.b = tf.zeros([out_neuron])
        self.output = tf.matmul(flatten_input,self.w)+self.b
        self.act_output = tf.nn.sigmoid(self.output)
        self.softmax_output = tf.nn.softmax(self.output)
#class Batch_Normalize(object):

class Load_Data():
    def __init__(self,image_size,RGB_layers):
        mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#        train_data, self.train_label = mnist.train.next_batch(batch_size)
        self.train_data = mnist.train.images.reshape([-1,image_size,image_size,RGB_layers])
        self.train_label = mnist.train.labels
        self.test_data= mnist.test.images.reshape([-1,image_size,image_size,RGB_layers])
        self.test_label = mnist.test.labels
def Training_Model(class_num,lr,batch_size,image_size,epochs):
    Data = Load_Data(image_size,1)
    CNN1 = CNN_Layers(image_size,batch_size,1,7,7,1,1)
    DNN1 = DNN_Layers(CNN1.pooling_conv, CNN1.pooling_conv_size_flatten, 256)
    DNN2 = DNN_Layers(DNN1.act_output,256,class_num)
    y_hat = tf.placeholder(tf.float32,[None, class_num])
    cross_entropy = -tf.reduce_sum(y_hat*tf.log(DNN2.softmax_output))
    f = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    print Data.train_label.shape[0]/batch_size
    for i in range(epochs):
        for j in range(Data.train_label.shape[0]/batch_size):
            CNN1_w_pre = CNN1.w.eval()
            sess.run(f,feed_dict={CNN1.x:Data.train_data[j*batch_size-1:j*batch_size+batch_size-1,:,:,:],
                                   y_hat:Data.train_label[j*batch_size-1:j*batch_size+batch_size-1,:]})

            CNN1_w_delta = tf.reduce_mean(tf.sub(CNN1_w_pre,CNN1.w))

            if j%10 == 0:
                correct_prediction = tf.equal(tf.argmax(DNN2.softmax_output,1), tf.argmax(y_hat,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print sess.run(accuracy, feed_dict={CNN1.x: Data.test_data, y_hat: Data.test_label})

if __name__ == '__main__':
    Training_Model(10,0.001,100,28,2)