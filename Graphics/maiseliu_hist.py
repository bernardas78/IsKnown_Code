from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
import os

# Reference image
img = Image.open("maiseliai.jpg")
img_rgb = np.asarray (img)
img_hsv = cv.cvtColor(img_rgb,cv.COLOR_RGB2HSV)


hist_img_maisas_rgb = np.vstack( [
        np.histogram ( img_rgb[:,:,0], bins=range(0,257) )[0] / np.sum(img_rgb[:,:,0]),
        np.histogram ( img_rgb[:,:,1], bins=range(0, 257))[0] / np.sum(img_rgb[:,:,1]),
        np.histogram ( img_rgb[:,:,2], bins=range(0, 257))[0] / np.sum(img_rgb[:,:,2])
    ])
hist_img_maisas_hsv = np.vstack( [
        np.histogram ( img_hsv[:,:,0], bins=range(0,257) )[0] / np.sum(img_hsv[:,:,0]),
        np.histogram ( img_hsv[:,:,1], bins=range(0, 257))[0] / np.sum(img_hsv[:,:,1]),
        np.histogram ( img_hsv[:,:,2], bins=range(0, 257))[0] / np.sum(img_hsv[:,:,2])
    ])

# Some bag-less images
cnt=0

# average histogram (RGB and HSV) holders
hist_rgb_bemaisu = np.zeros((3,256), dtype=float)
hist_hsv_bemaisu = np.zeros((3,256), dtype=float)

folder_bemaisu = r"BeMaisu"
for filename_bemaiso in os.listdir(folder_bemaisu):
    filename_full_bemaiso = os.path.join(folder_bemaisu,filename_bemaiso)
    img_bemaiso = Image.open(filename_full_bemaiso)
    img_bemaiso_rgb = np.asarray (img_bemaiso)

    hist_img_bemaiso_rgb = np.vstack( [
        np.histogram ( img_bemaiso_rgb[:,:,0], bins=range(0,257) )[0] / np.sum(img_bemaiso_rgb[:,:,0]),
        np.histogram ( img_bemaiso_rgb[:,:,1], bins=range(0, 257))[0] / np.sum(img_bemaiso_rgb[:,:,1]),
        np.histogram ( img_bemaiso_rgb[:,:,2], bins=range(0, 257))[0] / np.sum(img_bemaiso_rgb[:,:,2])
    ])
    hist_rgb_bemaisu = (hist_rgb_bemaisu*cnt + hist_img_bemaiso_rgb) / (cnt+1)

    img_bemaiso_hsv = cv.cvtColor(img_bemaiso_rgb,cv.COLOR_RGB2HSV)
    hist_img_bemaiso_hsv = np.vstack( [
        np.histogram ( img_bemaiso_hsv[:,:,0], bins=range(0,257) )[0] / np.sum(img_bemaiso_hsv[:,:,0]),
        np.histogram ( img_bemaiso_hsv[:,:,1], bins=range(0, 257))[0] / np.sum(img_bemaiso_hsv[:,:,1]),
        np.histogram ( img_bemaiso_hsv[:,:,2], bins=range(0, 257))[0] / np.sum(img_bemaiso_hsv[:,:,2])
    ])
    hist_hsv_bemaisu = (hist_hsv_bemaisu*cnt + hist_img_bemaiso_hsv) / (cnt+1)

    cnt +=1



# fig, axs = plt.subplots(3)
# fig.suptitle('Maiseliai, channel distribuition')
#
# axs[0].hist(img_rgb[:,:,0].ravel(), bins=255)
# axs[0].title.set_text("Red")
# axs[1].hist(img_rgb[:,:,1].ravel(), bins=255)
# axs[1].title.set_text("Green")
# axs[2].hist(img_rgb[:,:,2].ravel(), bins=255)
# axs[2].title.set_text("Blue")
# plt.show()
#
# fig, axs = plt.subplots(3)
# fig.suptitle('Maiseliai, HSV channel distribuition')
# axs[0].hist(img_hsv[:,:,0].ravel(), bins=255)
# axs[0].title.set_text("H")
# axs[1].hist(img_hsv[:,:,1].ravel(), bins=255)
# axs[1].title.set_text("S")
# axs[2].hist(img_hsv[:,:,2].ravel(), bins=255)
# axs[2].title.set_text("V")
# plt.show()

fig, axs = plt.subplots(2,3)

titles_rgb = ["R","G","B"]
titles_hsv = ["H","S","V"]
for i in range(3):
    axs[0,i].bar ( x=range(256), height=hist_img_bemaiso_rgb[i], width=1., color="blue", alpha=0.5, label="Be maiso")
    axs[0,i].bar ( x=range(256), height=hist_img_maisas_rgb[i], width=1., color="red", alpha=0.5, label="Maisas")
    axs[0,i].set_yticklabels("")
    axs[0,i].title.set_text("RGB " + titles_rgb[i])
    axs[0,i].legend()

    axs[1,i].bar ( x=range(256), height=hist_img_bemaiso_hsv[i], width=1., color="blue", alpha=0.5, label="Be maiso")
    axs[1,i].bar ( x=range(256), height=hist_img_maisas_hsv[i], width=1., color="red", alpha=0.5, label="Maisas")
    axs[1,i].set_yticklabels("")
    axs[1,i].title.set_text("HSV " + titles_hsv[i])
    axs[1,i].legend()

plt.show()