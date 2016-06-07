from model.basicCNN import *
from tool.load_cifar10 import *
from tensorflow.examples.tutorials.mnist import input_data
import time

if __name__=='__main__':
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
  net = modelX()
  net.loss(testData,testLabel)
  pass
