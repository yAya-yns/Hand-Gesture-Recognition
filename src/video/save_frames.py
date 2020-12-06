import cv2
import numpy as np

vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0



while success:

  image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

  cv2.imwrite(f'./frames/frame{count}.jpg', image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1