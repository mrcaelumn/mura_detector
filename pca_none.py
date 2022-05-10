#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
import random
import math
import os
import numpy as np
import time
from random import sample 
from sklearn.decomposition import PCA
import cv2


# In[ ]:


normal_train_img_fn = glob('mura_data/RGB/mura_march_clean/train_data/normal/*.png')


# In[ ]:


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
            if (y + windowSize[0]) > image.shape[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
                y_end_crop = True
            
            crop_x = x
            if (x + windowSize[1]) > image.shape[1]:
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


N,w=10000,256

def open_image(fn):
    img = cv2.imread(fn)
    # print(img.shape)
    img = sliding_window(img)
    # print(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, dsize=(w, w), interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(img)
    
    flat_b = np.array(b).ravel()
    flat_g = np.array(g).ravel()
    flat_r = np.array(r).ravel()
    return flat_b, flat_g, flat_r

def load_imgs(img_list):
    b_array = []
    g_array = []
    r_array = []
    
    for fn in img_list:
        flat_b, flat_g, flat_r = open_image(fn)
        b_array.append( np.array(flat_b).astype("float32") )
        g_array.append( np.array(flat_g).astype("float32") )
        r_array.append( np.array(flat_r).astype("float32") )
        
    
    return np.array(b_array), np.array(g_array), np.array(r_array)
    
    
# read_img = lambda fn: np.array(open_image(fn)).ravel()
# load_imgs = lambda fn_list: np.array([read_img(fn) for fn in fn_list]).astype("float32")


# In[ ]:


b_normal_train_imgs, g_normal_train_imgs, r_normal_train_imgs = load_imgs(sample(normal_train_img_fn,N))

print(b_normal_train_imgs.shape)
print(g_normal_train_imgs.shape)
print(r_normal_train_imgs.shape)


# In[ ]:


pca_b = PCA(n_components=256)
pca_g = PCA(n_components=256)
pca_r = PCA(n_components=256)

b_normal_train_PCA = pca_b.fit(b_normal_train_imgs)
g_normal_train_PCA = pca_g.fit(g_normal_train_imgs)
r_normal_train_PCA = pca_r.fit(r_normal_train_imgs)

print(b_normal_train_PCA)
print(g_normal_train_PCA)
print(r_normal_train_PCA)


# In[ ]:


def PCA_IMG(inputFile, outputFile, b_inner_pca, g_inner_pca, r_inner_pca):
    b, g, r = open_image(inputFile)
    # r_scaled = r / 255
    r_scaled = [r] 
    
    # g_scaled = g / 255
    g_scaled = [g] 
    
    # b_scaled = b / 255
    b_scaled = [b] 
    

    pca_r_trans = r_inner_pca.transform(r_scaled)

    pca_g_trans = g_inner_pca.transform(g_scaled)

    pca_b_trans = b_inner_pca.transform(b_scaled)

    pca_r_org = r_inner_pca.inverse_transform(pca_r_trans)
    pca_r_final = np.reshape(pca_r_org[0], (-1, w))
    
    # print(pca_r_final.shape)
    
    pca_g_org = g_inner_pca.inverse_transform(pca_g_trans)
    pca_g_final = np.reshape(pca_g_org[0], (-1, w))
    # print(pca_g_final.shape)
    
    pca_b_org = b_inner_pca.inverse_transform(pca_b_trans)
    pca_b_final = np.reshape(pca_b_org[0], (-1, w))
    # print(pca_b_final.shape)
    
    
    img_compressed = cv2.merge((pca_b_final, pca_g_final, pca_r_final))
    # print(img_compressed.shape)
    # print(img_compressed)
    # img = img_compressed * 255
    img = img_compressed
    cv2.imwrite(outputFile, img)
    # cv2.imwrite("test.png", img)
    


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
                PCA_IMG(inputFile, OutputFile, b_normal_train_PCA, g_normal_train_PCA, r_normal_train_PCA)
                
                # if index == 1:
                #     break
            if index % 1000 == 0:
                print("file: ",index)
        print("done.", class_name, mode)
        # if index == 1:
        #     break
    #     break

