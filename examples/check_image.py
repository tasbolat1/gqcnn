import cv2 
import numpy as np 
import matplotlib.pyplot as plt

a = np.load('./collect/orange_1.npy')
b = np.load('./collect/table_1.npy')
c = np.load('./collect/table_2.npy')

table_avg = 0.5*(b+c)
final_array = a-table_avg
final_array[np.abs(final_array) > 0.02] = 1
final_array[final_array != 1] = 0
print(final_array.shape)

fig, axs = plt.subplots(3)
axs[0].imshow(a)
axs[1].imshow(b)
axs[2].imshow(final_array)

plt.show()