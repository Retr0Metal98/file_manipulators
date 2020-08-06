'''
Module used to convert a series of image files into a single video file
'''
from cv2 import cv2
import numpy as np
import glob
import os

def conv_imgs_to_vid(img_folder_path,img_ext,vid_path,vid_fps=15):
    img_array = []
    img_count = 0
    for filename in glob.glob(img_path+"/**/*"+img_ext,recursive=True):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        img_count += 1
        print("Added {} images to array...".format(img_count), end='\r')

    print()
    _,vid_ext = os.path.splitext(vid_path)
    if vid_ext == '.avi':
        out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'DIVX'), vid_fps, size)
    
    print("Writing image array to "+vid_ext+" file...")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

