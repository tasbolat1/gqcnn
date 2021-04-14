import numpy as np
import cv2


img = cv2.imread("collect/left001.jpg",cv2.CV_16UC1)
# scale factor (realsense can detect up till 10m, but the rest 3m max)
SCALE_FACTOR = 1.0
print(img.shape)

# add one dimension at the end
#img = np.expand_dims(img, axis = -1)
cv2.imshow('asd',img)
cv2.waitKey(0)

#with open('./images/depth.npy', 'wb') as f:
#    np.save(f, img)

