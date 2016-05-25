#
# Don't know why save ones frame in the video but the mean is two
# Use print frame.mean() for checking
import numpy as np
import cv2
import time
import os

############
# Function #
############
def stateMachine(state, start_frame, end_frame, cap, frame, idx, path, name, pause):
    # output state, idx, process_flag, name, pause, item_id
    # state -1 : bulletproof
    if state == -1:
        if start_frame == -1:
            print 'start_frame not setting'
            return -1, idx, False, name, pause
        if end_frame == -1:
            print 'end_frame not setting'
            return -1, idx, False, name, pause
        return 0, idx, True, name, pause
    # state 0 : initialization(set frame & name)
    elif state == 0:
        name = raw_input('Enter image name: ')
        cap.set(1,start_frame)
        print 'Image Process Start...'
        return 1, idx, True, name, False
    # state 1 : image process or reset
    elif state == 1:
        if cap.get(1) > end_frame:
            cap.set(1,end_frame)
            start_frame = -1
            end_frame = -1
            idx = 0
            print 'Imager Process Finish...'
            print 'reset_frame',cap.get(1)
            return -1, idx, False, name, True
        else:
            idx += 1
            new_idx = idxGenerator(idx)
            folder = path + name
            #make folder
            if not os.path.isdir(folder):
                cmd = 'mkdir -p ' + folder
                os.system(cmd)
            fullname = folder + '/' + name + new_idx + '.png'
            cv2.imwrite(fullname,frame)
            return 1, idx, True, name, False

def auto_stateMachine(state,cap,frame,idx,path,item_id):
  if state == -1:
    if frame.mean() > 0 and frame.mean() < 5:
      item_id += 1
      print 'Image[' + str(item_id) + '] Process Start...'
      return 0, idx, item_id
    return -1, idx, item_id
  elif state == 0:
    if frame.mean() == 0:
      print 'Image[' + str(item_id) + '] Process finish...'
      idx = 0
      return -1, idx, item_id
    else:
      idx += 1
      new_idx = idxGenerator(idx)
      new_item_id = idxGenerator(item_id)
      folder = path + new_item_id
      #make folder
      if not os.path.isdir(folder):
        cmd = 'mkdir -p ' + folder
        os.system(cmd)
      fullname = folder + '/' + new_item_id + new_idx + '.png'
      cv2.imwrite(fullname,frame)
      return 0, idx, item_id

def idxGenerator(idx):
    # create id like 001 023 312
    d = idx / 10
    if d == 0:
        new_idx = '00' + str(idx)
    elif d > 0 and d < 9:
        new_idx = '0' + str(idx)
    else:
        new_idx = str(idx)
    return new_idx

def go(small=720):
  ############
  # Variable #
  ############
  cap = cv2.VideoCapture('../data/output.mov')

  width, height = int(cap.get(3)), int(cap.get(4))
  #small = min(width,height)
  total_frame = int(cap.get(7))
  fps = int(cap.get(5))
  pause = False
  end_flag = False
  process_flag = False
  start_frame = -1
  end_frame = -1
  state = -1
  name = None
  path = '../data/pic/'
  idx = 0
  item_id = 0

  #########
  # Start #
  #########
  start = time.time()
  while(True):
    # Capture frame-by-frame
    #if not pause:
    ret, frame = cap.read()

    current_frame = cap.get(1) #index of frame
    frame = cv2.resize(frame, (small, small))
    #print 'current_frame', current_frame
    # Record webcame

    # Prevent out of frame
    if current_frame == total_frame:
        break
    #    pause = True
    #    end_flag = True
    #else:
    #    end_flag = False

#    if process_flag:
    #state, idx, process_flag, name, pause, item_id = stateMachine(state,start_frame,end_frame,cap,frame,idx,path,name,pause,item_id)
    #print type(state), type(cap), type(frame), type(idx), type(path), type(item_id)
    #print state, cap, frame, idx, path, item_id
    #print frame.mean()
    state, idx, item_id = auto_stateMachine(state,cap,frame,idx,path,item_id)

    # show image
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF

    # keyboard interrupt
    if key == ord('q'): #and not process_flag:
        break
#    elif key == 32 and not end_flag and not process_flag:
#        pause = not pause
#        print 'PAUSE', pause
#    elif key == ord('s') and not process_flag:
#        start_frame = current_frame
#        print start_frame
#    elif key == ord('e') and not process_flag:
#        end_frame = current_frame - 1 #prevent last frame
#        print end_frame
#    elif key == ord('p'):
#        process_flag = not process_flag
#    elif key == ord('z'):
#        print '###########################'
#        print '#          STATUS         #'
#        print '###########################'
#        print 'PAUSE:', pause, 'End_Flag:', end_flag, 'Process_Flag:', process_flag
#        print 'Start_frame:' ,start_frame, 'End_frame:', end_frame,
#        print 'State: ', state
#    elif key == 2 and not process_flag: #leftkey jump back 10 frames
#        current_frame -= 10
#        if current_frame < 0: current_frame = 0
#        cap.set(1,current_frame)
#        print current_frame, cap.get(1)
#    elif key == 3 and not process_flag: #rightkey jump forward 10 frames
#        current_frame += 10
#        if current_frame > total_frame: current_frame = total_frame
#        cap.set(1,current_frame)
#        print current_frame, cap.get(1)
#    else: # show keyboard ID
#        if key != 255:
#            print key

  seconds = time.time() - start

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  pass
  #go(720)


# cap.get(id)
#id function note
#0 CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
#1 CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#2 CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
#3 CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4 CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#5 CV_CAP_PROP_FPS Frame rate.
#6 CV_CAP_PROP_FOURCC 4-character code of codec.
#7 CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#8 CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#9 CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#10 CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#11 CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#12 CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#13 CV_CAP_PROP_HUE Hue of the image (only for cameras).
#14 CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#15 CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#16 CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#17 CV_CAP_PROP_WHITE_BALANCE Currently not supported
#18 CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
