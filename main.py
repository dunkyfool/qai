from model.basicCNN import *
from tool.load_cifar10 import *
from tensorflow.examples.tutorials.mnist import input_data
import time

if __name__=='__main__':
  start = time.time()
  trainData,trainLabel,valData,valLabel,testData,testLabel = load(path='data/CIFAR-10')
  #print testData.shape, testLabel.shape
  print time.time()-start

  # speedup test
  #testData = testData[0:100,:]
  #testLabel = testLabel[0:100,:]

  trainData = trainData.reshape(-1,32,32,3)
  valData = valData.reshape(-1,32,32,3)
  testData = testData.reshape(-1,32,32,3)


  trainData = trainData-np.mean(trainData,axis=0)
  valData = valData-np.mean(trainData,axis=0)
  testData = testData-np.mean(testData,axis=0)
  net = modelX()
  start = time.time()
  net.loss(trainData,trainLabel,valData,valLabel,mode='train')
  print 'training over', time.time()-start
  net.loss(testData,testLabel,valData,valLabel,mode='test')
  pass
