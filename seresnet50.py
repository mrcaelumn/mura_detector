#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

# !pip install tf_clahe
# !pip install -U scikit-learn
# !pip install matplotlib
# !pip install pandas


# In[ ]:


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, MaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, Concatenate, Input, Dense, Reshape, Multiply, add, Flatten, ZeroPadding2D
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K


# In[ ]:


import itertools
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

import tf_clahe

import numpy as np
import pandas as pd 

from glob import glob
from tqdm import tqdm
from packaging import version
import os
import random
from packaging import version
from datetime import datetime
# Import writer class from csv module
from csv import DictWriter

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# new import
from tensorflow.keras.utils import Progbar
import time 

ORI_SIZE = (271, 481)
IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.
winSize = (256, 256)
stSize = 20

LIMIT_TRAIN_IMAGES = 5000
LIMIT_TEST_IMAGES = 200
EVAL_INTERVAL = 10

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
AUTOTUNE = tf.data.AUTOTUNE


# In[ ]:


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# In[ ]:


# class for SSIM loss function
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self,
         reduction=tf.keras.losses.Reduction.AUTO,
         name='SSIMLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, ori, recon):
        recon = tf.convert_to_tensor(recon)
        ori = tf.cast(ori, recon.dtype)

        loss_ssim = tf.reduce_mean(1 - tf.image.ssim(ori, recon, max_val=IMG_W, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))
        return loss_ssim

class MultiFeatureLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='FeatureLoss'):
        super().__init__(reduction=reduction, name=name)
        self.mse_func = tf.keras.losses.MeanSquaredError() 

    
    def call(self, real, fake, weight=1):
        result = 0.0
        for r, f in zip(real, fake):
            result = result + (weight * self.mse_func(r, f))
        
        return result


# In[ ]:


'''delcare all loss function that we will use'''

# for adversarial loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# L1 Loss
mae = tf.keras.losses.MeanAbsoluteError()

# L2 Loss
mse = tf.keras.losses.MeanSquaredError() 

multimse = MultiFeatureLoss()

# SSIM loss
ssim = SSIMLoss()


# In[ ]:


def enhance_image(image, beta=0.5):
    image = tf.cast(image, tf.float64)
    image = ((1 + beta) * image) + (-beta * tf.math.reduce_mean(image))
    return image
def custom_v3(img):

    img = tf.image.adjust_hue(img, 1.)
    img = tf.image.adjust_gamma(img)
    img = tfa.image.median_filter2d(img)
    return img

def crop_left_and_right(img, width=256, height=271):
    # img_shape = tf.shape(img)
    img_left = tf.image.crop_to_bounding_box(img, 0, 0, height, width)
    img_right = tf.image.crop_to_bounding_box(img, ORI_SIZE[0] - height, ORI_SIZE[1] - width, height, width)
    
    return [img_left, img_right]

def crop_left_and_right_select_one(img, width=256, height=271):
    # img_shape = tf.shape(img)
    img_left = tf.image.crop_to_bounding_box(img, 0, 0, height, width)
    img_right = tf.image.crop_to_bounding_box(img, ORI_SIZE[0] - height, ORI_SIZE[1] - width, height, width)
    if tf.math.reduce_std(img_left) < tf.math.reduce_std(img_right):
        return img_left
    return img_right

def sliding_crop_and_select_one(img, stepSize=stSize, windowSize=winSize):
    current_std = 0
    current_image = None
    y_end_crop, x_end_crop = False, False
    for y in range(0, ORI_SIZE[0], stepSize):
        
        y_end_crop = False
        
        for x in range(0, ORI_SIZE[1], stepSize):
            
            x_end_crop = False
            
            crop_y = y
            if (y + windowSize[0]) > ORI_SIZE[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
                y_end_crop = True
            
            crop_x = x
            if (x + windowSize[1]) > ORI_SIZE[1]:
                crop_x = ORI_SIZE[1] - windowSize[1]
                x_end_crop = True
                
            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])                
            std_image = tf.math.reduce_std(tf.cast(image, dtype=tf.float32))
          
            if current_std == 0 or std_image < current_std :
                current_std = std_image
                current_image = image
                
            if x_end_crop:
                break
                
        if x_end_crop and y_end_crop:
            break
            
    return current_image

def sliding_crop(img, stepSize=stSize, windowSize=winSize):
    current_std = 0
    current_image = []
    y_end_crop, x_end_crop = False, False
    for y in range(0, ORI_SIZE[0], stepSize):
        y_end_crop = False
        for x in range(0, ORI_SIZE[1], stepSize):
            x_end_crop = False
            crop_y = y
            if (y + windowSize[0]) > ORI_SIZE[0]:
                crop_y =  ORI_SIZE[0] - windowSize[0]
            
            crop_x = x
            if (x + windowSize[1]) > ORI_SIZE[1]:
                crop_x = ORI_SIZE[1] - windowSize[1]
            
            # print(crop_y, crop_x, windowSize)
            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])
            current_image.append(image)
            if x_end_crop:
                break
        if x_end_crop and y_end_crop:
            break
    return current_image


# In[ ]:


# function for  preprocessing data 
def prep_stage(x, training=True):
    beta_contrast = 0.1
    if training:
        
        x = enhance_image (x, beta_contrast)
        # x = custom_v3(x)
        
    else:
        x = enhance_image (x, beta_contrast)
        # x = custom_v3(x)
    
    return x

def post_stage(x):
    
    x = tf.image.resize(x, (IMG_H, IMG_W))
    # x = tf.image.random_crop(x, (IMG_H, IMG_W))
    # normalize to the range -1,1
    x = tf.cast(x, tf.float32)
    x = (x - 127.5) / 127.5
    # normalize to the range 0-1
    # img /= 255.0
    return x

def augment_dataset_batch_train(dataset_batch):

    flip_up_down = dataset_batch.map(lambda x: (tf.image.flip_up_down(x)), 
              num_parallel_calls=AUTOTUNE)
    
    flip_left_right = dataset_batch.map(lambda x: (tf.image.flip_left_right(x)), 
              num_parallel_calls=AUTOTUNE)
    
    dataset_batch = dataset_batch.concatenate(flip_up_down)
    dataset_batch = dataset_batch.concatenate(flip_left_right)
    
    
    return dataset_batch

def augment_dataset_batch_test(dataset_batch):
    AUTOTUNE = tf.data.AUTOTUNE
    
    
#     dataset_batch = dataset_batch.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
    
#     dataset_batch = dataset_batch.map(lambda x, y: (tf.image.per_image_standardization(x), y), 
#               num_parallel_calls=AUTOTUNE)
    
    
    return dataset_batch


# In[ ]:


def read_data_with_labels(filepath, class_names):
    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath,class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat
        path_list = []
        class_list = []
        for img in tqdm(os.listdir(path)):  
            if ".DS_Store" != img:
                filpath = os.path.join(path,img)
                
                path_list.append(filpath)
                
                class_list.append(class_num)
        
        n_samples = None
        if LIMIT_TEST_IMAGES != "MAX":
            n_samples = LIMIT_TEST_IMAGES
        path_list, class_list = shuffle(path_list, class_list, n_samples=n_samples ,random_state=random.randint(123, 10000))
        image_list = image_list + path_list
        label_list = label_list + class_list
    
    return image_list, label_list

# new function -> create image_list and filename_list
def read_data_with_filenames(filepath):
   
    image_list = []
    filename_list = []
    for img in os.listdir(filepath):  
        if ".DS_Store" != img:
            image_list.append(os.path.join(filepath,img))
            filename_list.append(img)
        
    image_list, filename_list = shuffle(image_list, filename_list, random_state=random.randint(123, 10000))
    
    # return image_list[:LIMIT_TRAIN_IMAGES], filename_list[:LIMIT_TRAIN_IMAGES]
    return image_list, filename_list




def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=IMG_C)
    # img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img, True)
    # img = crop_left_and_right_select_one(img)
    img = sliding_crop_and_select_one(img, )
    img = post_stage(img)

    return img

def load_image_with_label(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=IMG_C)
    # img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img, False)
    # img = crop_left_and_right(img)

    img_list = sliding_crop(img)
    img = [post_stage(a) for a in img_list]
    
    # img = post_stage(img)
    # return l_img, r_img, label
    return img, label

# new -> create dataset with filename
def tf_dataset(images_path, batch_size, labels=False, class_names=None):
    
    images_path = shuffle(images_path, random_state=random.randint(123, 10000))
    
    if LIMIT_TRAIN_IMAGES != "MAX":
        images_path = images_path[:LIMIT_TRAIN_IMAGES]
        
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    
    # tf.size(dataset)
    # dataset = dataset.shuffle(buffer_size=512, seed=random.randint(123, 10000))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def tf_dataset_labels(images_path, batch_size, class_names=None):
    
    filenames, labels = read_data_with_labels(images_path, class_names)
   
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=512, seed=random.randint(123, 10000))
    
    dataset = dataset.map(load_image_with_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


# In[ ]:


# load image dataset for testing with labels
def load_image_test(filename, class_names, size=(IMG_H,IMG_W)):
	# load image with the preferred size
    pixels = tf_dataset_labels(images_path=filename, batch_size=1, class_names=class_names)
    pixels = augment_dataset_batch_test(pixels)
    
    return pixels

# load image dataset for trainnig without labels
def load_image_train(filename, batch_size):
	# load image with the preferred size
    
    pixels = tf_dataset(filename, batch_size)
    
    pixels = augment_dataset_batch_train(pixels)

    return pixels


# In[ ]:


def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name_model+'_roc_curve.png')
    plt.show()
    plt.clf()
    


''' calculate the auc value for lables and scores'''
def roc(labels, scores, name_model):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    # print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshod that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_threshold

def plot_loss_with_rlabel(x_value, y_value, real_label, name_model, prefix, label_axis=["x_label", "y_label"]):
    # 'bo-' means blue color, round points, solid lines
    colours = ["blue" if x == 1.0 else "red" for x in real_label]
    plt.scatter(x_value, y_value, label='loss_value',c = colours)
#     plt.rcParams["figure.figsize"] = (50,3)
    # Set a title of the current axes.
    plt.title(prefix + "_" + name_model)
    # show a legend on the plot
    red_patch = mpatches.Patch(color='red', label='Normal Display')
    blue_patch = mpatches.Patch(color='blue', label='Defect Display')
    plt.legend(handles=[red_patch, blue_patch])
    # Display a figure.
    plt.xlabel(label_axis[0])
    plt.ylabel(label_axis[1])
    
    plt.savefig(name_model + "_" + prefix +'_rec_feat_rlabel.png')
    plt.show()
    plt.clf()


# In[ ]:


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'_cm.png')
    plt.show()
    plt.clf()


# In[ ]:


def conv_block_2nd(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_2nd(x, num_filters)
    return x


# In[ ]:


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
        
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name


    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = add([x, input_tensor], name='block_' + block_name + '_x4')
    x = Activation('relu', name='block_out_' + block_name + '_x4')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name

    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)
    
    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])
    
    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '_prj')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_prj_bn')(shortcut)

    x = add([x, shortcut], name='block_' + block_name)
    x = Activation('relu', name='block_out_' + block_name)(x)
    return x


def SEResNet50(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=225,
                                      min_size=160,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name="input_1")
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, name="input_1")
        else:
            img_input = input_tensor
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    # x = ZeroPadding2D(padding=(2, 2), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='conv1_pool')(x)
    # x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # x = Flatten()(x)
    # x = Dense(classes, activation='softmax', name='fc6')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='se-resnet50')
    return model  


# In[ ]:


def build_seresnet50_unet(input_shape):
    inputs = Input(input_shape, name="input_1")
    """ Pre-trained ResNet50 Model """
    seresnet50 = SEResNet50(weights=None, input_tensor=inputs)
    # seresnet50.summary()
    """ Encoder """
    s1 = seresnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = seresnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = seresnet50.get_layer("relu3_1_x1").output  ## (128 x 128)
    s4 = seresnet50.get_layer("relu4_1_x1").output  ## (64 x 64)
    s5 = seresnet50.get_layer("relu5_1_x1").output  ## (32 x 32)

    """ Bridge """
    b1 = seresnet50.get_layer("block_out_5_3_x4").output  ## (16 x 16)

    """ Decoder """
    x = IMG_SIZE
    d1 = decoder_block(b1, s5, x)                     ## (32 x 32)
    x = x/2
    d2 = decoder_block(d1, s4, x)                     ## (64 x 64)
    x = x/2
    d3 = decoder_block(d2, s3, x)                     ## (128 x 128)
    x = x/2
    d4 = decoder_block(d3, s2, x)                      ## (256 x 256)
    x = x/2
    d5 = decoder_block(d4, s1, x)                      ## (512 x 512)


    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="tanh")(d4)

    model = Model(inputs, outputs, name="SEResNet50_U-Net")
    return model


# In[ ]:


# create discriminator model
def build_discriminator(inputs):
    num_layers = 4
    f = [2**i for i in range(num_layers)]
    x = inputs
    features = []
    for i in range(0, num_layers):
        if i == 0:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (4, 4), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * IMG_H ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        else:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (4, 4), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * IMG_H ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        
        features.append(x)
        
        
        
    
    x = tf.keras.layers.Flatten()(x)
    features.append(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.models.Model(inputs, outputs = [features, output])
    
    return model


# In[ ]:


class ResUnetGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator):
        super(ResUnetGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
       
        # Regularization Rate for each loss function
        self.ADV_REG_RATE_LF = 1
        self.REC_REG_RATE_LF = 50
        self.SSIM_REG_RATE_LF = 10
        self.FEAT_REG_RATE_LF = 1
        self.field_names = ['epoch', 'gen_loss', 'disc_loss']
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
    
    
    def compile(self, g_optimizer, d_optimizer, filepath, resume=False):
        super(ResUnetGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
#         columns name (epoch, gen_loss, disc_loss)
        
        
        logs = pd.DataFrame([], columns=self.field_names)

        if not resume:
            logs.to_csv(filepath, encoding='utf-8', index=False)
        else:
            fileExist = os.path.exists(filepath)
            if not fileExist:
                print("file not found. then we create new file")
                logs.to_csv(filepath, encoding='utf-8', index=False)
            

            
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print("Images: ", images)
            reconstructed_images = self.generator(images, training=True)
            feature_real, label_real = self.discriminator(images, training=True)
            # print(generated_images.shape)
            feature_fake, label_fake = self.discriminator(reconstructed_images, training=True)
            # Loss 1: ADVERSARIAL loss
            
            real_loss = cross_entropy(label_real, tf.ones_like(label_real))
            fake_loss = cross_entropy(label_fake, tf.zeros_like(label_fake))
            disc_adv_loss = real_loss + fake_loss
            
            gen_adv_loss = cross_entropy(label_fake, tf.ones_like(label_real))
            
            # Loss 2: RECONSTRUCTION loss (L1)
            loss_rec = mae(images, reconstructed_images)
        
            # Loss 3: SSIM Loss
            loss_ssim =  ssim(images, reconstructed_images)
        
            # Loss 4: FEATURE Loss
            # loss_feat = mse(feature_real, feature_fake)
            loss_feat = multimse(feature_real, feature_fake)
            
            gen_loss = tf.reduce_mean( 
                (gen_adv_loss * self.ADV_REG_RATE_LF) 
                + (loss_rec * self.REC_REG_RATE_LF) 
                + (loss_feat * self.FEAT_REG_RATE_LF) 
                + (loss_ssim * self.SSIM_REG_RATE_LF) 
            )
            
            disc_loss = tf.reduce_mean( (disc_adv_loss * self.ADV_REG_RATE_LF) + (loss_feat * self.FEAT_REG_RATE_LF) )


        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        

        
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        


        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "gen_adv_loss": gen_adv_loss,
            "disc_adv_loss": disc_adv_loss,
            "loss_rec": loss_rec,
            "loss_ssim": loss_ssim,
            "loss_feat": loss_feat
        }

    def saved_model(self, gmodelpath, dmodelpath):
        self.generator.save(gmodelpath)
        self.discriminator.save(dmodelpath)

    def loaded_model(self, g_filepath, d_filepath):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        
    # load and save data of training process
    def load_save_processing(self,filepath, epoch_num, disc_loss, gen_loss, g_filepath, d_filepath, resume=False):
        # columns name (epoch, gen_loss, disc_loss)

        if resume:
            # load logs data
            logs = pd.read_csv(filepath)
            logs.sort_values("epoch", ascending=False)
            epoch = max(logs['epoch'].tolist(), default=0)
            
            epoch_list = logs['epoch'].tolist()
            disc_loss = logs['disc_loss'].tolist()
            gen_loss = logs['gen_loss'].tolist()
            
            
            # load model data
            self.loaded_model(g_filepath, d_filepath)
            print(epoch, disc_loss, gen_loss)
            return epoch, epoch_list, disc_loss, gen_loss
        
        else:
            data={'epoch':epoch_num,'disc_loss':disc_loss,'gen_loss':gen_loss}
            print("row added." , data)
            f_object = open(filepath, "a+")
            dwriter = DictWriter(f_object, fieldnames=self.field_names)
            dwriter.writerow(data)
            f_object.close()
            return None, None, None, None
            
    def calculate_a_score(self, images):
        
        anomaly_weight = 0.7
        
        reconstructed_images = self.generator(images, training=False)
        # images = grayscale_converter(images)
        feature_real, label_real  = self.discriminator(images, training=False)
        # print(generated_images.shape)
        feature_fake, label_fake = self.discriminator(reconstructed_images, training=False)

        # Loss 2: RECONSTRUCTION loss (L1)
        loss_rec = mae(images, reconstructed_images)

        loss_feat = multimse(feature_real, feature_fake)
        # print("loss_rec:", loss_rec, "loss_feat:", loss_feat)
        score = loss_rec + (anomaly_weight * loss_feat)
        return score, loss_rec, loss_feat
    
    def testing(self, test_dateset, g_filepath, d_filepath, name_model, evaluate=False):
        
        scores_ano = []
        real_label = []
        rec_loss_list = []
        feat_loss_list = []
        # ssim_loss_list = []
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        
        counter = 0
        for images, labels in test_dateset:
            loss_rec, loss_feat = 0.0, 0.0
            score = 0
            counter += 1

            '''for normal'''
            # temp_score, loss_rec, loss_feat = self.calculate_a_score(images)
            # score = temp_score.numpy()


            '''for sliding images'''
            for image in images:
                r_score, r_rec_loss, r_feat_loss = self.calculate_a_score(image)
                if r_score.numpy() > score or score == 0:
                    score = r_score.numpy()
                    loss_rec = r_rec_loss
                    loss_feat = r_feat_loss
                
            scores_ano = np.append(scores_ano, score)
            real_label = np.append(real_label, labels.numpy()[0])
        
            rec_loss_list = np.append(rec_loss_list, loss_rec)
            feat_loss_list = np.append(feat_loss_list, loss_feat)
            
            if (counter % 100) == 0:
                print(counter, " tested.")
        
        
        ''' Scale scores vector between [0, 1]'''
        scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
        label_axis = ["recon_loss", "scores_anomaly"]
        plot_loss_with_rlabel(rec_loss_list, scores_ano, real_label, name_model, "anomaly_score", label_axis)
        # print("scores_ano: ", scores_ano)
        # print("real_label: ", real_label)
#         scores_ano = (scores_ano > threshold).astype(int)
        auc_out, threshold = roc(real_label, scores_ano, name_model)
        if evaluate:
            return auc_out
        
        print("auc: ", auc_out)
        print("threshold: ", threshold)
        
        
        
        scores_ano = (scores_ano >= threshold).astype(int)
        cm = tf.math.confusion_matrix(labels=real_label, predictions=scores_ano).numpy()
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[0][0]
        print(cm)
        print(
                "model saved. TP %d:, FP=%d, FN=%d, TN=%d" % (TP, FP, FN, TN)
        )
        plot_confusion_matrix(cm, class_names, title=name_model)
        # label_axis = ["ssim_loss", "recon_loss"]
        # plot_loss_with_rlabel(ssim_loss_list, rec_loss_list, real_label, name_model, "recontruction_loss", label_axis)
        
        diagonal_sum = cm.trace()
        sum_of_all_elements = cm.sum()

        print("Accuracy: ", diagonal_sum / sum_of_all_elements )
        print("False Alarm Rate (FPR): ", FP/(FP+TN))
        print("Leakage Rat (FNR): ", FN/(FN+TP))
        print("TNR: ", TN/(FP+TN))
        print("precision_score: ", TP/(TP+FP))
        # print("recall_score (func): ", recall_score(real_label, scores_ano))
        print("recall_score (manual): ", TP/(TP+FN))
        print("NPV: ", TN/(FN+TN))
#         F1 = 2 * (precision * recall) / (precision + recall)
        print("F1-Score: ", f1_score(real_label, scores_ano))
    
    
    
        
    def checking_gen_disc(self, mode, g_filepath, d_filepath, test_data_path):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
#         path = "mura_data/RGB/test_data/normal/normal.bmp"
#         path = "mura_data/RGB/test_data/defect/defect.bmp"
#         path = "rgb_serius_defect/BUTTERFLY (2).bmp"
        paths = {
            "normal": test_data_path+"/normal/normal.png",
            "defect": test_data_path+"/defect/defect.png",
        }
   
        for i, v in paths.items():
            print(i,v)
            
            width=IMG_W
            height=IMG_H
            rows = 1
            cols = 3
            axes=[]
            fig = plt.figure()
            
            
            img = tf.io.read_file(v)
            img = tf.io.decode_bmp(img, channels=IMG_C)
            
            name_subplot = mode+'_original_'+i
            axes.append( fig.add_subplot(rows, cols, 1) )
            axes[-1].set_title('_original_')  
            plt.imshow(img.numpy().astype("int64"), alpha=1.0)
            plt.axis('off')
#             plt.savefig(name_original+'.png')
        
        
        
            img = prep_stage(img)
            img = tf.cast(img, tf.float64)

            name_subplot = mode+'_preprocessing_'+i
            axes.append( fig.add_subplot(rows, cols, 2) )
            axes[-1].set_title('_preprocessing_')  
            plt.imshow(img.numpy().astype("int64"), alpha=1.0)
            plt.axis('off')
            img = (img - 127.5) / 127.5
#             plt.savefig(mode+'_preprocessing_'+i+'.png')
   
        
            image = tf.reshape(img, (-1, IMG_H, IMG_W, IMG_C))
            reconstructed_images = self.generator.predict(image)
            reconstructed_images = tf.reshape(reconstructed_images, (IMG_H, IMG_W, IMG_C))
#             reconstructed_images = reconstructed_images[0, :, :, 0] * 127.5 + 127.5
#             reconstructed_images = reconstructed_images[0]
            reconstructed_images = reconstructed_images * 127 + 127

            name_subplot = mode+'_reconstructed_'+i
            axes.append( fig.add_subplot(rows, cols, 3) )
            axes[-1].set_title('_reconstructed_') 
            plt.imshow(reconstructed_images.numpy().astype("int64"), alpha=1.0)
            plt.axis('off')
            
            fig.tight_layout()    
            fig.savefig(mode+'_'+i+'.png')
            plt.show()
            plt.clf()


# In[ ]:


def plot_epoch_result(epochs, loss, name, model_name, colour):
        plt.plot(epochs, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(model_name+ '_'+name+'_epoch_result.png')
        plt.show()
        plt.clf()
        
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self,
                 g_model_path,
                 d_model_path,
                 logs_file,
                 name_model
                ):
        super(CustomSaver, self).__init__()
        self.g_model_path = g_model_path
        self.d_model_path = d_model_path
        self.logs_file = logs_file
        self.name_model = name_model
        self.epochs_list = []
        self.gen_loss_list = []
        self.disc_loss_list = []
        
    
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}
            
    def on_train_end(self, logs=None):
        self.model.saved_model(self.g_model_path, self.d_model_path)
        
        self.plot_epoch_result(self.epochs_list, self.gen_loss_list, "Generator_Loss", self.name_model, "g")
        self.plot_epoch_result(self.epochs_list, self.disc_loss_list, "Discriminator_Loss", self.name_model, "r")
    
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
#             print(k, v)
            self.history.setdefault(k, []).append(v)
        
        self.epochs_list.append(epoch)
        self.gen_loss_list.append(logs["gen_loss"])
        self.disc_loss_list.append(logs["disc_loss"])
        
        
        self.model.load_save_processing(logs_file, epoch, logs["disc_loss"], logs["gen_loss"], self.g_model_path, self.d_model_path, resume=False) 
        
        if (epoch + 1) % 15 == 0 or (epoch + 1) <= 15:
            self.model.saved_model(self.g_model_path, self.d_model_path)
            print('saved for epoch',epoch + 1)
            
    def plot_epoch_result(self, epochs, loss, name, model_name, colour):
        plt.plot(epochs, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(model_name+ '_'+name+'_epoch_result.png')
        plt.show()
        plt.clf()

        
def scheduler(epoch, lr):
    if epoch < 1500:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def set_callbacks(name_model, logs_path, logs_file, path_gmodal, path_dmodal):
    # create and use callback:
    
    saver_callback = CustomSaver(
        path_gmodal,
        path_dmodal,
        logs_file,
        name_model
    )
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='disc_loss', factor=0.2,
                              patience=7, min_lr=0.000001)
    
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=logs_path + name_model + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"), 
#         histogram_freq=1
#     )
    

    callbacks = [
        saver_callback,
#         checkpoints_callback,
        # tensorboard_callback,
#         lr_callback,
        # reduce_lr,
    ]
    return callbacks


# In[ ]:


def run_trainning(model, train_dataset, num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps, resume=False, ):
    init_epoch = 0
    
    epochs_list = []
    gen_loss_list = []
    disc_loss_list = []
    auc_score = 0.6
    auc_score_std = 0.7
    # callbacks = set_callbacks(name_model, logs_path, logs_file, path_gmodal, path_dmodal)
    # if resume:
    #     print("resuming trainning. ", name_model)
    #     skip_epoch, _, _, _ = model.load_save_processing(logs_file, num_epochs, [], [], path_gmodal, path_dmodal, resume=resume)
    #     if skip_epoch < num_epochs:
    #         init_epoch = skip_epoch
    class_names = ["normal", "defect"] # normal = 0, defect = 1
    test_dateset = load_image_test(test_data_path, class_names)
    
    for epoch in range(0, num_epochs):
        epoch += 1
        print("running epoch: ", epoch)
        # final_dataset = train_dataset.shuffle(buffer_size=3, seed=123, reshuffle_each_iteration=True).take(steps)
        
        result = model.fit(
            train_dataset, 
            epochs = 1,
                  # epochs=num_epochs, 
                  # callbacks=callbacks, 
                  # initial_epoch=init_epoch,
            # shuffle=True, 
            # steps_per_epoch=steps
        )
        
        # if result["gen_loss"].numpy() <= 1000:
        #     epochs_list.append(epoch)
        #     gen_loss_list.append(result["gen_loss"])
        #     disc_loss_list.append(result["disc_loss"])
            
        
        epochs_list.append(epoch)
        gen_loss_list.append(result.history["gen_loss"][0])
        disc_loss_list.append(result.history["disc_loss"][0])
        
        if epoch % 10 == 0 or epoch >= 10 or epoch == num_epochs:
            model.saved_model(path_gmodal, path_dmodal)
            print('saved for epoch:', epoch)
        
        if epoch % EVAL_INTERVAL == 0 and epoch >= EVAL_INTERVAL:
            auc = model.testing(test_dateset, path_gmodal, path_dmodal, name_model, evaluate=True)
            print(
                    "model evaluated at epoch %d: with AUC=%f" % (epoch, auc)
                )
            if auc > auc_score or auc > auc_score_std:
                
                best_g_model_path = path_gmodal.replace(".h5", f"_best_{epoch}_{auc:.2f}.h5")
                best_d_model_path = path_dmodal.replace(".h5", f"_best_{epoch}_{auc:.2f}.h5")
                
                model.saved_model(best_g_model_path, best_d_model_path)
                auc_score = auc
                print(
                    "the best model saved. at epoch %d: with AUC=%f" % (epoch, auc)
                )
    
    plot_epoch_result(epochs_list, gen_loss_list, "Generator_Loss", name_model, "g")
    plot_epoch_result(epochs_list, disc_loss_list, "Discriminator_Loss", name_model, "r")


# In[ ]:


if __name__ == "__main__":
    
    '''
    In Default:
    Clahe: OFF
    BCET: OFF
    Resize: crop or padding (decided by tensorflow)
    Datasets: For trainning dataset, it'll have additional datasets (flip-up-down and flip-right-left)
    '''
    
    # run the function here
    """ Set Hyperparameters """
    
    mode = "seresnet_20220407"
    colour = "RGB" # RGB & GS (GrayScale)
    batch_size = 32
    steps = 160
    num_epochs = 500
    
    name_model= f"{str(IMG_H)}_{colour}_{mode}_{str(num_epochs)}_{str(LIMIT_TRAIN_IMAGES)}"
    
    resume_trainning = False
    lr = 0.0001
    
    print("start: ", name_model)
    
    # set dir of files
    train_images_path = f"mura_data/{colour}/mura_march_clean/train_data/normal/*.png"
    test_data_path = f"mura_data/{colour}/mura_march_clean/test_data"
    saved_model_path = f"mura_data/{colour}/saved_model/"
    
    logs_path = f"mura_data/{colour}/logs/"
    
    logs_file = logs_path + "logs_" + name_model + ".csv"
    
    path_gmodal = saved_model_path + name_model + "_g_model" + ".h5"
    path_dmodal = saved_model_path +  name_model + "_d_model" + ".h5"
    
    """
    Create a MirroredStrategy object. 
    This will handle distribution and provide a context manager (MirroredStrategy.scope) 
    to build your model inside.
    """
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    # print(input_shape)
    
    ## init models ##
    
    # set input 
    inputs = tf.keras.layers.Input(input_shape, name="input_1")
    
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    g_model = build_seresnet50_unet(input_shape)

    d_model = build_discriminator(inputs)

#     d_model.summary()
#     g_model.summary()

    resunetgan = ResUnetGAN(g_model, d_model)
    resunetgan.compile(g_optimizer, d_optimizer, logs_file, resume_trainning)
    
    """ run trainning process """
    train_images = glob(train_images_path)
    train_images_dataset = load_image_train(train_images, batch_size)
    train_images_dataset = train_images_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    run_trainning(resunetgan, train_images_dataset, num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps, resume=resume_trainning)
    
    """ run testing """
    class_names = ["normal", "defect"] # normal = 0, defect = 1
    test_dateset = load_image_test(test_data_path, class_names)
    resunetgan.testing(test_dateset, path_gmodal, path_dmodal, name_model)
    # resunetgan.checking_gen_disc(mode, path_gmodal, path_dmodal, test_data_path)


# In[ ]:




