#DEPENDENCIES

#if use linux run these commands

# $  sudo apt-get install python3-opencv
# $  pip3 install opencv

#if you use windows run this command

# python -m pip install opencv-python



import cv2

vidcap = cv2.VideoCapture('your_video_here.mp4') #add you own video file here
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(f"frame{count}.jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
