import numpy as np
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

  # save hdf5 file
  #with

def load():
  pass

if __name__ == '__main__':
  convert()
  pass
