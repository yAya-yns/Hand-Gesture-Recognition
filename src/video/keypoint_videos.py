import cv2
import numpy as np
import os

num_files = len(os.listdir('./labelled_frames')) - 20

with open('./detected_gestures.npy', 'rb') as f:
    gestures = np.load(f, allow_pickle=True)

img_array = []
for i in range(num_files):

    buffer = gestures[max(0, i-7):min(i+7, num_files)]
    if len(buffer) < 5:
        gesture = None
    else:
        gesture = max(set(buffer), key=list(buffer).count)

    label = f'Detected gesture is {gesture}'
    img = cv2.imread(f'./labelled_frames/frame{i}.jpg.png')

    img = cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()