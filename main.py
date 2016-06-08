from model.basicCNN import *
from tool.load_cifar10 import *
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt

def quick_scan(X,y,X1,y1,lr_range=[-3.0,-3.7],reg_range=[1,0]):
  results = {}
  learning_rates = lr_range
  regularization_strengths = reg_range
  best_val = -1

  tic = time.time()
  for i in range(10):
    print '['+str(i)+']'
    # random choose lr & reg within the range
    lr = 10**np.random.uniform(learning_rates[0],learning_rates[1])
    reg = 10**np.random.uniform(regularization_strengths[0],regularization_strengths[1])
    print 'lr:\t'+str(lr)
    print 'reg:\t'+str(reg)

    net = modelX()
    net.loss(X,y,X1,y1,mode='train',lr=lr,reg=reg,batch=100,epoch=1)
    results[(lr,reg)]=(net.X_acc_history[-1],net.X1_acc_history[-1])
    if best_val < net.X1_acc_history[-1]:
      best_val = net.X1_acc_history[-1]

  toc = time.time()
  print 'Total Training: computed in %fs' % (toc - tic)
  print 'Best Validation Record %.5f' % (best_val)

  ##########################################
  # Visualize the cross-validation results #
  ##########################################
  import math
  x_scatter = [math.log10(x[0]) for x in results]
  y_scatter = [math.log10(x[1]) for x in results]

  # plot training accuracy
  marker_size = 100
  colors = [results[x][0] for x in results]
  plt.subplot(2, 1, 1)
  plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
  plt.colorbar()
  plt.xlabel('log learning rate')
  plt.ylabel('log regularization strength')
  plt.title('CIFAR-10 training accuracy')

  # plot validation accuracy
  colors = [results[x][1] for x in results] # default size of markers is 20
  plt.subplot(2, 1, 2)
  plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
  plt.colorbar()
  plt.xlabel('log learning rate')
  plt.ylabel('log regularization strength')
  plt.title('CIFAR-10 validation accuracy')
  plt.show()


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
  #net = modelX()
  #start = time.time()
  quick_scan(trainData,trainLabel,valData,valLabel)
  #net.loss(trainData,trainLabel,valData,valLabel,mode='train',batch=100)
  #print 'training over', time.time()-start
  #net.loss(testData,testLabel,valData,valLabel,mode='test')
  pass
