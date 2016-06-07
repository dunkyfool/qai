from model.basicCNN import *
from tool.load_cifar10 import *
from tensorflow.examples.tutorials.mnist import input_data
import time

if __name__=='__main__':
  '''
  start = time.time()
  _,_,_,_,testData,testLabel = load(path='data/CIFAR-10')
  print testData.shape, testLabel.shape
  print time.time()-start

  # speedup test
  testData = testData[0:100,:]
  testLabel = testLabel[0:100,:]

  testData = testData.reshape(-1,32,32,3)

  _,H,W,C = testData.shape
  _,cls = testLabel.shape

  testData = testData-np.mean(testData,axis=0)
  '''
  mnist = input_data.read_data_sets("archive/MNIST_data/",one_hot=True)
  #train_data = mnist.train.images.reshape([-1,28,28,1])
  #train_label = mnist.train.labels
  test_data = mnist.test.images.reshape([-1,28,28,1])
  test_label = mnist.test.labels

  #print train_data.shape, train_label.shape
  #print test_data.shape, test_label.shape

  testData = test_data[0:100,:]
  testLabel = test_label[0:100,:]
  testData -= testData.mean(0)
  #print type(testData)

  net = modelX()
  net.loss(testData,testLabel)

  #_,H,W,C = testData.shape
  #_,cls = testLabel.shape

  #net = model1(input_dim=(H,W,C),num_class=cls,hidden=[500],batch=2)
  #net.loss(testData,testLabel,'train',lr=1e-2,ep=100)

  pass
