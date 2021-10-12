#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import tf_clahe
import os
print(tf.__version__)


# In[ ]:


def bcet(img):
    Lmin = tf.reduce_min(img) # MINIMUM OF INPUT IMAGE
#     Lmin = np.min(img) # MINIMUM OF INPUT IMAGE
#     print("Lmin", Lmin)
    Lmax = tf.reduce_max(img) # MAXIMUM OF INPUT IMAGE
#     Lmax = np.max(img) # MAXIMUM OF INPUT IMAGE
#     print("Lmax", Lmax)
    Lmean = tf.reduce_mean(img) #MEAN OF INPUT IMAGE
#     Lmean = np.mean(img) #MEAN OF INPUT IMAGE
#     print("Lmean", Lmean)
    LMssum = tf.reduce_mean(img * img) #MEAN SQUARE SUM OF INPUT IMAGE
#     LMssum = np.mean(pow(img, 2)) #MEAN SQUARE SUM OF INPUT IMAGE
#     print("LMssum", LMssum)

    Gmin = tf.constant(0, dtype="float32") #MINIMUM OF OUTPUT IMAGE
    Gmax = tf.constant(255, dtype="float32") #MAXIMUM OF OUTPUT IMAGE
    Gmean = tf.constant(110, dtype="float32") #MEAN OF OUTPUT IMAGE
    
    subber = tf.constant(2, dtype="float32")
    
    # find b
    
    bnum = ((Lmax**subber)*(Gmean-Gmin)) - (LMssum*(Gmax-Gmin)) + ((Lmin**subber) *(Gmax-Gmean))
    bden = subber * ((Lmax*(Gmean-Gmin)) - (Lmean*(Gmax-Gmin)) + (Lmin*(Gmax-Gmean)))
    
    b = bnum/bden
    
    # find a
    a1 = Gmax-Gmin
    a2 = Lmax-Lmin
    a3 = Lmax+Lmin-(subber*b)
            
    a = a1/(a2*a3)
    
    # find c
    c = Gmin - (a*(Lmin-b)**subber)
    
    # Process raster
    y = a*((img - b)**subber) + c #PARABOLIC FUNCTION
    
    final = tf.clip_by_value(y, clip_value_min=0, clip_value_max=255)
    return y


# In[ ]:


def bcet_processing(img,channels=3):
#     img = tf.make_tensor_proto(img,dtype="int64")
#     img = tf.make_ndarray(img)
#     print(img.shape)
    layers = []
    for i in range(channels):
        layer = img[:,:,i]
        layer = bcet(layer)
        layers.append(layer)
        
# #     print(red.shape)
#     blue = img[:,:,1]
# #     print(blue.shape)
#     green = img[:,:,2]
#     red = img[:,:,0]
# #     print(red.shape)
#     blue = img[:,:,1]
# #     print(blue.shape)
#     green = img[:,:,2]
# #     print(green.shape)
    
#     red = bcet(red)
# #     print(red.shape)
#     blue = bcet(blue)
# #     print(blue.shape)
#     green = bcet(green)
# #     print(green.shape)
    
#     final_image = np.stack((red, blue, green), axis=-1)
#     final_image = tf.convert_to_tensor(final_image, dtype=tf.int64) 
#     print(final_image.shape)
    final_image = tf.stack(layers, axis=-1)
#     print(final_image.shape)
    return final_image


# In[ ]:




def convert_file(inputFile, outputFile):

    img = tf.io.read_file(inputFile)
    img = tf.io.decode_bmp(img, channels=3)
    # print(tf.rank(img))
    img = tf.cast(img, tf.float32)

    # img = tf_clahe.clahe(img)
    
    img = bcet_processing(img)
    
    tf.keras.utils.save_img(outputFile, img)
#     plt.figure()
#     img = tf.cast(img, tf.int32)
#     plt.imshow(img)
# #     # plt.savefig('normal_bcet.png')
#     plt.savefig(outputFile, format="jpg")   # save the figure to file
#     plt.close() 


# In[ ]:


## convert colour of images

# projects/mura_detector/Series Defect Upload_20210809/BUTTERFLY
# projects/mura_detector/BCET_Photo/serious defect/butterfly


Input_dir = 'prod_data/RGB/train_data/normal/'
Out_dir = 'prod_data/RGB/train_data/bcet_normal/'
a = os.listdir(Input_dir)
for i in a:
    if i != ".DS_Store":
        print(i)
        inputFile = Input_dir+i
        OutputFile = Out_dir+i
        convert_file(inputFile, OutputFile)
    
#     break


# In[ ]:




