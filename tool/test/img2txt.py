from os import listdir, system
from os.path import isfile, join, isdir
import h5py
import numpy as np
import cv2
import time



#############
# Save file #
#############
def saveFile(onlyfolders,path,txt_path):
    """
    Convert all images in ../data/pic/$img into ../txt/XXX.h5
    onlyfolders: list all folder in ../data
    """
    start = time.time()
    img = None
    print 'Image convert to file start...'
    for folder in onlyfolders:
        #list all files in folder & remove .DS_Store
        filename = [f for f in listdir(join(path,folder)) if isfile(join(path,folder,f))]
        if '.DS_Store' in filename:
          filename.remove('.DS_Store')
        #print imgs
        txtname = txt_path + '/' + folder + '.h5'
        with h5py.File(txtname, 'w') as hf:
            for name in filename:
                img = cv2.imread(join(path,folder,name),0)
                hf.create_dataset(name,data=img.reshape(1,-1))
            #save picture's shape in each record
            shape = img.shape
            if len(shape) == 2:
              shape = shape + (1,)
            print shape
            hf.create_dataset('shape',data=shape)
    print 'Image convert to file finish...'
    print 'Time:', time.time()-start

#############
# Load file #
#############
def loadFile(txt_path):
  """
  Load all ../txt/XXX.h5 & transfer into train(dictionary)
  train['data'] = np.shape(all,feature)
  train['label'] = np.shape(all)
  """
  start = time.time()
  print 'Load data start...'
  train = {}
  trainData = []
  trainLabel = []
  mark = []
  shape = None
  label = None

  #list all record in ../txt & remove .DS_Store
  filename = [f for f in listdir(txt_path) if isfile(join(txt_path,f))]
  if '.DS_Store' in filename:
    filename.remove('.DS_Store')
  print filename

  # load pixel and assign label & reshape it as picture
  for label, name in enumerate(filename):
    txtname = join(txt_path,name)
    with h5py.File(txtname,'r') as hf:
      print hf.keys()
      shape = np.array(hf.get('shape'))
      for i in hf.keys():
        if i != 'shape':
          trainData += [np.array(hf.get(i))]
          mark += [label]

  # shape = (-1,H,W,C)
  shape = np.insert(shape,0,-1)
  #print np.asarray(trainData).shape

  # build label (N,cls)
  trainLabel = np.zeros((len(mark),label+1))
  for i in range(len(mark)):
    trainLabel[i,mark[i]] += 1

  train['data'] = np.asarray(trainData).reshape(shape)
  train['label'] = trainLabel
  print 'Load data finish...'
  print 'Time:', time.time() - start

  for k, v in sorted(train.iteritems()):
    print k, v.shape
  return train

def go():
  ############
  # Variable #
  ############
  path = '../data/pic'
  txt_path = '../txt'

  # check pic folder
  if not isdir(path):
    cmd = 'mkdir -p ' + path
    system(cmd)

  # check txt folder
  if not isdir(txt_path):
    cmd = 'mkdir -p ' + txt_path
    system(cmd)

  # list folder in data
  onlyfolders = [f for f in listdir(path) if isdir(join(path, f))]
  #print onlyfolders
  saveFile(onlyfolders,path,txt_path)
  trainDic = loadFile(txt_path)
  #return trainDic

########
# main #
########
if __name__ == '__main__':
  pass
  #go()
  #saveFile(onlyfolders)
  #trainDic = loadFile(txt_path)

