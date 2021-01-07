import numpy as np
import cv2


img = cv2.imread("images/depth.png",cv2.CV_16UC1)
# scale factor (realsense can detect up till 10m, but the rest 3m max)
SCALE_FACTOR = 1.0
img = img*SCALE_FACTOR

#img = img/65535.0
img = img/1000.0
# add one dimension at the end
img = np.expand_dims(img, axis = -1)

with open('./images/depth.npy', 'wb') as f:
    np.save(f, img)

