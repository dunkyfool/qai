from model.basicCNN import *
from tool.load_cifar10 import *
import time
import matplotlib.pyplot as plt

def quick_scan(X,y,X1,y1,lr_range=[-3.0,-3.7],reg_range=[1,0]):
  results = {}
  learning_rates = lr_range
  regularization_strengths = reg_range
  best_val = -1
  best_lr, best_reg = None, None

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
    results[(lr,reg)]=(max(net.X_acc_history),max(net.X1_acc_history))
    if best_val < max(net.X1_acc_history):
      best_val = max(net.X1_acc_history)
      best_lr = lr
      best_reg = reg

  toc = time.time()
  print 'Total Training: computed in %fs' % (toc - tic)
  print 'Best Validation Record %.5f' % (best_val)
  print 'Best Validation learning rate %.10f' % (best_lr)
  print 'Best Validation regularization %.10f' % (best_reg)

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

def marathon(X,y,X1,y1,X2,y2,lr=1e-4,reg=1e-4,epoch=20):
  net = modelX()
  net.loss(X,y,X1,y1,mode='train',lr=lr,reg=reg,batch=100,epoch=epoch)
  net.loss(X2,y2,X1,y1,mode='test')

  ####################################################
  # Visualize training loss and train / val accuracy #
  ####################################################
  plt.subplot(2, 1, 1)
  plt.title('Training loss')
  plt.plot(net.X_loss_history, 'o')
  plt.xlabel('Iteration')

  plt.subplot(2, 1, 2)
  plt.title('Accuracy')
  plt.plot(net.X_acc_history, '-o', label='train')
  plt.plot(net.X1_acc_history, '-o', label='val')
  plt.plot([0.5] * len(net.X1_acc_history), 'k--')
  plt.xlabel('Epoch')
  plt.legend(loc='lower right')
  plt.gcf().set_size_inches(15, 12)
  plt.show()
  pass


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

  quick_scan(trainData,trainLabel,valData,valLabel,lr_range=[-2.7,-3.4],reg_range=[0.9,0.2])
  #marathon(trainData,trainLabel,valData,valLabel,testData,testLabel)
  pass
