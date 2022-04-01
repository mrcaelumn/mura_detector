#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

# !pip install tf_clahe
# !pip install -U scikit-learn
# !pip install matplotlib
# !pip install pandas


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

IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.

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

        loss_ssim = tf.math.reduce_sum(1 - tf.image.ssim(ori, recon, 2.0))
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
    img = tf.cast(img, tf.float64)
    img = tfio.experimental.color.rgb_to_bgr(img)
    img = tf.image.adjust_contrast(img, 11.)
    img = tf.image.adjust_hue(img, 11.)
    img = tf.image.adjust_gamma(img)
    img = tfa.image.median_filter2d(img)
    return img
    


# In[ ]:


# function for  preprocessing data 
def prep_stage(x, training=True):
    beta_contrast = 0.1
    if training:
        # x = tf.image.adjust_gamma(x, gamma=1, gain=1)
        # x = tfa.image.median_filter2d(x, filter_shape=(3, 3))
        # x = custom_v3(x)
        x = enhance_image (x, beta_contrast)
        
    else:
        # x = tf.image.adjust_gamma(x, gamma=1, gain=1)
        # x = tfa.image.median_filter2d(x, filter_shape=(3, 3))
        # x = tf.image.sobel_edges(x)
        # x = custom_v3(x)
        x = enhance_image (x, beta_contrast)
    
    x = tf.image.resize(x, (IMG_H, IMG_W))
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
#                 print(filpath, class_num)
                
                path_list.append(filpath)
                class_list.append(class_num)
                # image_label_list.append({filpath:class_num})
        
        path_list, class_list = shuffle(path_list, class_list, random_state=random.randint(123, 10000))
        image_list = image_list + path_list[:LIMIT_TEST_IMAGES]
        label_list = label_list + class_list[:LIMIT_TEST_IMAGES]
  
    # print(image_list, label_list)
    
    return image_list, label_list


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img)
    img = tf.cast(img, tf.float32)
#     rescailing image from 0,255 to -1,1
    img = (img - 127.5) / 127.5
    
    return img

def load_image_with_label(image_path, label):
#     print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img, training=False)
    img = tf.cast(img, tf.float32)
    #     rescailing image from 0,255 to -1,1
    img = (img - 127.5) / 127.5
    # img /= 255.0
    
    return img, label


def tf_dataset(images_path, batch_size, labels=False, class_names=None):
    
    if LIMIT_TRAIN_IMAGES != "MAX":
        images_path = images_path[:LIMIT_TRAIN_IMAGES]
        
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    # tf.size(dataset)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def tf_dataset_labels(images_path, batch_size, class_names=None):
    
    filenames, labels = read_data_with_labels(images_path, class_names)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=10240)
    
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


def conv_block(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (3, 3), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# In[ ]:


# create generator model based on resnet50 and unet network
def build_generator_resnet50_unet(input_shape):
    # print(inputs)
    # print("pretained start")
    """ Pre-trained ResNet50 Model """
    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_shape)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (256 x 256)
    s2 = resnet50.get_layer("conv1_relu").output        ## (128 x 128)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (64 x 64)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (32 x 32)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (16 x 16)

    """ Decoder """
    x = IMG_H
    d1 = decoder_block(b1, s4, x)                     ## (32 x 32)
    x = x/2
    d2 = decoder_block(d1, s3, x)                     ## (64 x 64)
    x = x/2
    d3 = decoder_block(d2, s2, x)                     ## (128 x 128)
    x = x/2
    d4 = decoder_block(d3, s1, x)                      ## (256 x 256)
    
    """ Output """
#     outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d4)
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="tanh")(d4)
#     outputs = tf.keras.layers.Conv2D(3, 1, padding="same")(d5)

    model = tf.keras.models.Model(inputs, outputs)

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
            x = tf.keras.layers.SeparableConv2D(f[i] * IMG_H ,kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        else:
            x = tf.keras.layers.SeparableConv2D(f[i] * IMG_H ,kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            # x = tf.keras.layers.Dropout(0.3)(x)
        
        features.append(x)
        
    # feature = x
    x = tf.keras.layers.SeparableConv2D(1, (3, 3), padding='valid', use_bias=False)(x)
    features.append(x)
    x = tf.keras.layers.Flatten()(x)
    # features.append(x)
    # output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    output = tf.keras.layers.Activation('sigmoid')(x)

    
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
            
            
    def testing(self, test_dateset, g_filepath, d_filepath, name_model, evaluate=False):

        anomaly_weight = 0.1
        
        scores_ano = []
        real_label = []
        rec_loss_list = []
        feat_loss_list = []
        # ssim_loss_list = []
        i = 0
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        
        
        for images, labels in test_dateset:
            i += 1
            
            reconstructed_images = self.generator(images, training=False)
            feature_real, label_real  = self.discriminator(images, training=False)
            # print(generated_images.shape)
            feature_fake, label_fake = self.discriminator(reconstructed_images, training=False)

            
           
            loss_rec = mae(images, reconstructed_images)
        
            # loss_feat = mse(feature_real, feature_fake)
            loss_feat = multimse(feature_real, feature_fake)
            
            score = (anomaly_weight * loss_rec) + ((1-anomaly_weight) * loss_feat)
          
            scores_ano = np.append(scores_ano, score.numpy())
            real_label = np.append(real_label, labels.numpy()[0])
        
            rec_loss_list = np.append(rec_loss_list, loss_rec)
            feat_loss_list = np.append(feat_loss_list, loss_feat)
        
        
        ''' Scale scores vector between [0, 1]'''
        scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
        label_axis = ["recon_loss", "scores_anomaly"]
        plot_loss_with_rlabel(rec_loss_list, scores_ano, real_label, name_model, "anomaly_score", label_axis)
#         print("scores_ano: ", scores_ano)
#         print("real_label: ", real_label)
#         scores_ano = (scores_ano > threshold).astype(int)
        auc_out, threshold = roc(real_label, scores_ano, name_model)
        if evaluate:
            return auc_out
        
        # print("auc: ", auc_out)
        # print("threshold: ", threshold)
        
        
        
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


def run_trainning(model, train_dataset,num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps,resume=False):
    init_epoch = 0
    
    epochs_list = []
    gen_loss_list = []
    disc_loss_list = []
    auc_score = 0.0
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
        
        epochs_list.append(epoch)
        gen_loss_list.append(result.history["gen_loss"][0])
        disc_loss_list.append(result.history["disc_loss"][0])
        
        # print(epochs_list)
        # print(gen_loss_list)
        # print(disc_loss_list)
        
        if epoch % 10 == 0 or epoch >= 10 or epoch == num_epochs:
            model.saved_model(path_gmodal, path_dmodal)
            print('saved for epoch:', epoch)
        
        if epoch % EVAL_INTERVAL == 0 and epoch >= EVAL_INTERVAL:
            auc = model.testing(test_dateset, path_gmodal, path_dmodal, name_model, evaluate=True)
            print(
                    "model evaluated at epoch %d: with AUC=%f" % (epoch, auc)
                )
            if auc > auc_score:
                
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
    
    mode = "20220210"
    colour = "RGB" # RGB & GS (GrayScale)
    batch_size = 32
    steps = 160
    num_epochs = 150
    
    name_model= f"{str(IMG_H)}_{colour}_{mode}_{str(num_epochs)}_{str(LIMIT_TRAIN_IMAGES)}"
    
    resume_trainning = False
    lr = 0.0001
    
    print("start: ", name_model)
    
    # set dir of files
    train_images_path = f"mura_data/{colour}/mura_clean/train_data/normal/*.png"
    test_data_path = f"mura_data/{colour}/mura_clean/test_data"
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
    g_model = build_generator_resnet50_unet(inputs)

    d_model = build_discriminator(inputs)

#     d_model.summary()
#     g_model.summary()

    resunetgan = ResUnetGAN(g_model, d_model)
    resunetgan.compile(g_optimizer, d_optimizer, logs_file, resume_trainning)
    
    """ run trainning process """
    train_images = glob(train_images_path)
    # size_of_dataset = len(list(train_images_dataset)) * batch_size
    # batch_size = int(len(train_images)/steps)
    # print("batch_size: ", batch_size)
    train_images_dataset = load_image_train(train_images, batch_size)
    train_images_dataset = train_images_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    run_trainning(resunetgan, train_images_dataset, num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps,resume=resume_trainning)
    
    """ run testing """
    class_names = ["normal", "defect"] # normal = 0, defect = 1
    test_dateset = load_image_test(test_data_path, class_names)
    resunetgan.testing(test_dateset, path_gmodal, path_dmodal, name_model)
    # resunetgan.checking_gen_disc(mode, path_gmodal, path_dmodal, test_data_path)


# In[ ]:




