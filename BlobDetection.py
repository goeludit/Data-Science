from skimage.feature import blob_dog, blob_log  , blob_doh
from math import sqrt 
from matplotlib import pyplot as plt 
from skimage.io import imread 

from skimage.color import rgb2gray
from matplotlib import cm 
image = imread("C:\\Users\\Udit\\Desktop\\wint_sky.gif", as_gray= True)


blob_logs = blob_log(image, threshold=.1, max_sigma = 30 , num_sigma=10)

number_rows = len(blob_logs)
print(number_rows)
fig, ax = plt.subplots(1,1)
plt.imshow(image) 
for blob in blob_logs:
    y,x,r = blob
    c = plt.Circle((x,y), radius = r+5, color = 'lime', linewidth = 2, fill = False)
    ax.add_patch(c)
plt.show()
