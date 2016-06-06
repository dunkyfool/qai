import numpy as np
import cv2
import h5py
from os import system
from os.path import isdir

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#/Users/dunkyfool/QAI/data/CIFAR-10/cifar-10-batches-py
def convert():
  # load all batches 
  addr1 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_1'
  dict1 = unpickle(addr1)
  addr2 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_2'
  dict2 = unpickle(addr2)
  addr3 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_3'
  dict3 = unpickle(addr3)
  addr4 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_4'
  dict4 = unpickle(addr4)
  addr5 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_5'
  dict5 = unpickle(addr5)

  # check key and value in the dictionary
  '''
  print '#1'
  for key, value in dict1.iteritems() :
      print key, type(value)
  print '#2'
  for key, value in dict2.iteritems() :
      print key, type(value)
  print '#3'
  for key, value in dict3.iteritems() :
      print key, type(value)
  print '#4'
  for key, value in dict4.iteritems() :
      print key, type(value)
  print '#5'
  for key, value in dict5.iteritems() :
      print key, type(value)
  '''

  # show the amount of the data & label
  '''
  print '#1'
  print dict1['data'].shape
  print len(dict1['labels'])
  print '#2'
  print dict2['data'].shape
  print len(dict2['labels'])
  print '#3'
  print dict3['data'].shape
  print len(dict3['labels'])
  print '#4'
  print dict4['data'].shape
  print len(dict4['labels'])
  print '#5'
  print dict5['data'].shape
  print len(dict5['labels'])
  '''

  # concatenate all datas
  data = np.concatenate([dict1['data'],
                         dict2['data'],
                         dict3['data'],
                         dict4['data'],
                         dict5['data']],axis=0)
  print data.shape

  # concatenate all labels 
  _mask = np.eye(10)
  _1 = np.array([_mask[idx,:] for idx in dict1['labels']]).reshape(-1,10)
  _2 = np.array([_mask[idx,:] for idx in dict2['labels']]).reshape(-1,10)
  _3 = np.array([_mask[idx,:] for idx in dict3['labels']]).reshape(-1,10)
  _4 = np.array([_mask[idx,:] for idx in dict4['labels']]).reshape(-1,10)
  _5 = np.array([_mask[idx,:] for idx in dict5['labels']]).reshape(-1,10)
  label = np.concatenate([_1,_2,_3,_4,_5],axis=0)
  print label.shape

  # review whether the data/label is correct
  '''
  print 'check data...'
  print 'data #1',np.array_equal(data[0:10000,:],dict1['data'])
  print 'data #2',np.array_equal(data[10000:20000,:],dict2['data'])
  print 'data #3',np.array_equal(data[20000:30000,:],dict3['data'])
  print 'data #4',np.array_equal(data[30000:40000,:],dict4['data'])
  print 'data #5',np.array_equal(data[40000:50000,:],dict5['data'])

  print 'check label...'
  print 'label #1',np.array_equal(label[0:10000].argmax(1),np.array(dict1['labels']))
  print 'label #2',np.array_equal(label[10000:20000].argmax(1),np.array(dict2['labels']))
  print 'label #3',np.array_equal(label[20000:30000].argmax(1),np.array(dict3['labels']))
  print 'label #4',np.array_equal(label[30000:40000].argmax(1),np.array(dict4['labels']))
  print 'label #5',np.array_equal(label[40000:50000].argmax(1),np.array(dict5['labels']))
  '''

  # check folder exist
  path = '../data/CIFAR-10'
  if not isdir(path):
    cmd = 'mkdir -p '+path
    system(cmd)

  # reshape/transpose image pixel (N,H,W,C) 
  data = data.reshape(-1,3,32,32)
  data = data.transpose(0,2,3,1)
  data = data.reshape(-1,3*32*32)

  # check image format
  '''
  for i in range(10,20,1):
    img = data[i]
    #img = img.reshape(3,32,32)
    #img = img.transpose(1,2,0)
    img = img.reshape(32,32,3)
    img = cv2.resize(img,(256,256))
    while(True):
      cv2.imshow('show',img)
      key = cv2.waitKey(1) & 0xFF
      if key==ord('q'): break
    cv2.destroyAllWindows()
  '''

  # train/val/test
  trainData = data[0:40000,:]
  trainLabel = label[0:40000,:]
  valData = data[40000:45000,:]
  valLabel = label[40000:45000,:]
  testData = data[45000:50000,:]
  testLabel = label[45000:50000,:]

  #print trainData.shape, trainLabel.shape
  #print valData.shape, valLabel.shape
  #print testData.shape, testLabel.shape

  # save hdf5 file
  txtname = path+'/cifar10.h5'

  x = raw_input('Remove old data[y/n]')
  if x=='n': pass
  else:
    cmd = 'rm '+txtname
    system(cmd)

  with h5py.File(txtname, 'w') as hf:
    #hf.create_dataset(name,data=img.reshape(1,-1))
    hf.create_dataset('trainData',data=trainData,compression='gzip')
    hf.create_dataset('trainLabel',data=trainLabel,compression='gzip')
    hf.create_dataset('valData',data=valData,compression='gzip')
    hf.create_dataset('valLabel',data=valLabel,compression='gzip')
    hf.create_dataset('testData',data=testData,compression='gzip')
    hf.create_dataset('testLabel',data=testLabel,compression='gzip')

def load():
  path = '../data/CIFAR-10'
  txtname = path+'/cifar10.h5'
  trainData, trainLabel, valData, valLabel, testData, testLabel = None, None, None, None, None, None
  with h5py.File(txtname,'r') as hf:

    #for key,value in hf.iteritems():
    #  print key, value.shape

    trainData = np.array(hf['trainData'])
    trainLabel = np.array(hf['trainLabel'])
    valData = np.array(hf['valData'])
    valLabel = np.array(hf['valLabel'])
    testData = np.array(hf['testData'])
    testLabel = np.array(hf['testLabel'])

    #print trainData.shape, trainLabel.shape
    #print valData.shape, valLabel.shape
    #print testData.shape, testLabel.shape


    #########################
    # check data is correct #
    #########################
    '''
    # load all batches 
    addr1 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_1'
    dict1 = unpickle(addr1)
    addr2 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_2'
    dict2 = unpickle(addr2)
    addr3 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_3'
    dict3 = unpickle(addr3)
    addr4 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_4'
    dict4 = unpickle(addr4)
    addr5 = '../data/CIFAR-10/cifar-10-batches-py/data_batch_5'
    dict5 = unpickle(addr5)

    # concatenate all datas
    data = np.concatenate([dict1['data'],
                         dict2['data'],
                         dict3['data'],
                         dict4['data'],
                         dict5['data']],axis=0)
    #print data.shape

    # concatenate all labels 
    _mask = np.eye(10)
    _1 = np.array([_mask[idx,:] for idx in dict1['labels']]).reshape(-1,10)
    _2 = np.array([_mask[idx,:] for idx in dict2['labels']]).reshape(-1,10)
    _3 = np.array([_mask[idx,:] for idx in dict3['labels']]).reshape(-1,10)
    _4 = np.array([_mask[idx,:] for idx in dict4['labels']]).reshape(-1,10)
    _5 = np.array([_mask[idx,:] for idx in dict5['labels']]).reshape(-1,10)
    label = np.concatenate([_1,_2,_3,_4,_5],axis=0)
    #print label.shape

    print type(trainData), trainData.shape
    trainData = trainData.reshape(-1,32,32,3)
    trainData = trainData.transpose(0,3,1,2)
    trainData = trainData.reshape(-1,3*32*32)
    print np.array_equal(trainData, data[0:40000,:])
    print np.array_equal(trainLabel, label[0:40000,:])

    print type(valData), valData.shape
    valData = valData.reshape(-1,32,32,3)
    valData = valData.transpose(0,3,1,2)
    valData = valData.reshape(-1,3*32*32)
    print np.array_equal(valData, data[40000:45000,:])
    print np.array_equal(valLabel, label[40000:45000,:])

    print type(testData), testData.shape
    testData = testData.reshape(-1,32,32,3)
    testData = testData.transpose(0,3,1,2)
    testData = testData.reshape(-1,3*32*32)
    print np.array_equal(testData, data[45000:50000,:])
    print np.array_equal(testLabel, label[45000:50000,:])
    '''
  pass

if __name__ == '__main__':
  #convert()
  load()
  pass
