#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
import random
import math
import os
import numpy as np
from PIL import Image
import time
from random import sample 
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import cv2
import tensorflow as tf
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# In[ ]:


sob_normal_train_img_fn = glob('mura_data/RGB/mura_march_clean/train_data/normal/*.png')


# In[ ]:


ORI_SIZE = (271, 481)

def sliding_window(image, stepSize=20, windowSize=(256, 256)):
    current_std = 0
    current_image = None
    # print(image.shape)
    y_end_crop, x_end_crop = False, False
    
    for y in range(0, image.shape[0], stepSize):
        y_end_crop = False
        
        for x in range(0, image.shape[1], stepSize):
            
            x_end_crop = False
            
            crop_y = y
            if (y + windowSize[0]) > ORI_SIZE[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
                y_end_crop = True
            
            crop_x = x
            if (x + windowSize[1]) > ORI_SIZE[1]:
                crop_x = ORI_SIZE[1] - windowSize[1]
                x_end_crop = True
            
            # print(x, y)
            img = image[crop_y:y + windowSize[0], crop_x:x + windowSize[1]]
            std_image = np.std(img)
            # print(std_image)
            if current_std == 0 or std_image < current_std :
                current_std = std_image
                current_image = img
            
            if x_end_crop:
                break
                
        if x_end_crop and y_end_crop:
            break
            
    return current_image   


# In[ ]:


N,w=300,128

def open_image(fn):
    img = cv2.imread(fn)
    
    img = sliding_window(img)
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(w, w), interpolation=cv2.INTER_CUBIC)
    return img
    
read_img = lambda fn: np.array(open_image(fn)).ravel()
load_imgs = lambda fn_list: np.array([read_img(fn) for fn in fn_list]).astype("float32")


# In[ ]:


sob_normal_train_imgs = load_imgs(sample(sob_normal_train_img_fn,N))

print(sob_normal_train_imgs.shape)


# In[ ]:


pca = PCA(n_components=256)
sob_normal_train_PCA = pca.fit(sob_normal_train_imgs)
print(sob_normal_train_PCA)


# In[ ]:


def PCA_IMG(inputFile, outputFile, inner_pca):
    img = np.array(open_image(inputFile)).ravel().astype("float32")
    x_pca = inner_pca.transform([img])
    x_inv = inner_pca.inverse_transform(x_pca)

    img = np.reshape(x_inv[0], (-1, w))

    # print(img.shape)
    # cv2.imwrite(OutputFile, img)
    cv2.imwrite("test.png", img)    


# In[ ]:


## convert colour of images
for mode in ["test_data","train_data"]:
    for class_name in ["normal", "defect"]:
        Input_dir = f'mura_data/RGB/mura_march_clean/{mode}/{class_name}/'
        Out_dir = f'mura_data/RGB/mura_pca_clean/{mode}/{class_name}/'
        a = os.listdir(Input_dir)
        index = 0
        for i in a:
            index += 1
            if i != ".DS_Store" and i != ".ipynb_checkpoints":

                inputFile = Input_dir+i
                OutputFile = Out_dir+i
                PCA_IMG(inputFile, OutputFile, pca)
                
                # if index == 1:
                #     break
            if index % 1000 == 0:
                print("file: ",index)
        print("done.", class_name, mode)
        # if index == 1:
        #     break
    #     break


# In[ ]:




