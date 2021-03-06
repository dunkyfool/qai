# Save gray scale video with swith control
#
# FPS default:28, grayscale:21, add mode: 24, save_video: 12, save_image_13
# gray2rgb: repeat each channel
# output.avi is a fps:10 gray scale video
#
import numpy as np
import cv2
import time
import os

###########
# Get FPS #
###########
def getfps(flag, current_time, _fps):
  if flag==1 and current_time==0:
    current_time = time.time()
    _fps += 1
  elif flag==1 and current_time!=0:
    _fps +=1
  elif flag==0 and current_time!=0:
    seconds = time.time() - current_time
    print 'fps:', _fps / seconds
    current_time = 0
    _fps = 0
  return current_time, _fps

########
# main #
########
def go():
  ############
  # Variable #
  ############
  cap = cv2.VideoCapture(0)
  #cap = cv2.VideoCapture('vtest.avi')
  path = '../data/pic'
  if not os.path.isdir(path):
    cmd = 'mkdir -p ' + path
    os.system(cmd)

  fps = 10
  width, height = int(cap.get(3)), int(cap.get(4))
  capSize = (width, height)
  fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
  vout = cv2.VideoWriter()
  success = vout.open('../data/output.mov',fourcc,fps,capSize)
  mode = False
  current_time = 0 # record start time 
  _fps = 0         # frame counter
  start_frame = np.ones((height,width,3),dtype=np.uint8)  # for auto-split[start]
  end_frame = np.zeros((height,width,3),dtype=np.uint8)   # for auto-split[end]

  #########
  # Start #
  #########
  start = time.time()
  vout.write(start_frame)
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    _fps +=1
    # Turn frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.repeat(frame.reshape(1,-1),3,axis=0)
    frame = frame.reshape((3,height,width)).transpose(1,2,0)
    #print frame.shape, type(frame), type(frame[0,0,0])
    # Record webcame
    if mode:
        vout.write(frame)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not mode:
          #print start_frame.shape, type(start_frame), type(start_frame[0,0,0])
          vout.write(start_frame)
        else:
          vout.write(end_frame)
        mode = not mode
        print 'Record', mode

  seconds = time.time() - start
  print _fps / seconds

  # When everything done, release the capture
  vout.release()
  vout = None
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  pass
  #go()
