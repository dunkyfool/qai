def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def convert():
  dict = unpickle('../CIFAR-10/cifar-10-batches-py/data_batch_1')
  for key, value in dict.iteritems() :
      print key, type(value)
  print dict['data'].shape
  print len(dict['labels'])
  print dict['batch_label']
  print len(dict['filenames'])
  for i in range(10):
    print dict['labels'][i]
    print dict['filenames'][i]

if __name__ == '__main__':
  convert()
  pass
