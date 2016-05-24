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
    """
    start = time.time()
    print 'Image convert to file start...'
    for folder in onlyfolders:
        filename = [f for f in listdir(join(path,folder)) if isfile(join(path,folder,f))]
        if '.DS_Store' in filename:
          filename.remove('.DS_Store')
        #print imgs
        txtname = txt_path + '/' + folder + '.h5'
        with h5py.File(txtname, 'w') as hf:
            for name in filename:
                img = cv2.imread(join(path,folder,name),0)
                hf.create_dataset(name,data=img.reshape(1,-1))
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
  feature = None
  filename = [f for f in listdir(txt_path) if isfile(join(txt_path,f))]
  if '.DS_Store' in filename:
    filename.remove('.DS_Store')
  print filename
  for label, name in enumerate(filename):
    txtname = join(txt_path,name)
    with h5py.File(txtname,'r') as hf:
      print hf.keys()
      for i in hf.keys():
        feature = np.array(hf.get(i)).shape[1]
        trainData += [np.array(hf.get(i))]
        trainLabel += [label]
  train['data'] = np.asarray(trainData).reshape(-1,feature)
  train['label'] = np.asarray(trainLabel)
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
  return trainDic

########
# main #
########
if __name__ == '__main__':
  pass
  #saveFile(onlyfolders)
  #trainDic = loadFile(txt_path)

