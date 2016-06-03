import webcam
import imgPP
import img2txt
from os import system

output = raw_input('Clear old data[y/n]: ')
if output == 'y':
  cmd1 = 'rm -rf ../data/'
  cmd2 = 'rm -rf ../txt'
  system(cmd1)
  system(cmd2)

webcam.go()
imgPP.go(small=48)
trainDic = img2txt.go()
