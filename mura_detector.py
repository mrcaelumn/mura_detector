#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

get_ipython().system('pip install tf_clahe')
get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')


# In[ ]:


import itertools
import tensorflow as tf
import numpy as np
import pandas as pd 
import tf_clahe
from glob import glob
from tqdm import tqdm
import time
import os
from datetime import datetime
# Import writer class from csv module
from csv import DictWriter

from sklearn.metrics import roc_curve, auc, precision_score, recall_score

from matplotlib import pyplot as plt

IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.


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

        # Loss 3: SSIM Loss
        loss_ssim =  tf.reduce_mean(1 - tf.image.ssim(ori, recon, max_val=1.0)[0]) 

        return loss_ssim


# In[ ]:


# class for Feature loss function
class FeatureLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='FeatureLoss'):
        super().__init__(reduction=reduction, name=name)

    
    def call(self, real, fake):
        fake = tf.convert_to_tensor(fake)
        real = tf.cast(real, fake.dtype)
        # Loss 4: FEATURE Loss
        loss_feat = tf.reduce_mean(tf.pow((real-fake), 2))
        return loss_feat


# In[ ]:


# class for Adversarial loss function
class AdversarialLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='AdversarialLoss'):
        super().__init__(reduction=reduction, name=name)

    
    def call(self, logits_in, labels_in):
        labels_in = tf.convert_to_tensor(labels_in)
        logits_in = tf.cast(logits_in, labels_in.dtype)
        # Loss 4: FEATURE Loss
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))
        


# In[ ]:


# delcare all loss function that we will use

# for adversarial loss
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
cross_entropy = AdversarialLoss()
# L1 Loss
mae = tf.keras.losses.MeanAbsoluteError()
# L2 Loss
mse = tf.keras.losses.MeanSquaredError() 
feat = FeatureLoss()

# SSIM loss
ssim = SSIMLoss()


# In[ ]:


# function for  preprocessing data 
def prep_stage(x):
    x = tf_clahe.clahe(x)
    x = tf.image.resize_with_pad(x, IMG_H, IMG_W)
    return x

def augment_dataset_batch_train(dataset_batch):
    AUTOTUNE = tf.data.AUTOTUNE
    
   
    dataset_batch = dataset_batch.map(lambda x: (tf.image.per_image_standardization(x)))
        
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
    
    dataset_batch = dataset_batch.map(lambda x, y: (tf.image.per_image_standardization(x), y), 
              num_parallel_calls=AUTOTUNE)
    
    
    return dataset_batch


# In[ ]:


def read_data_with_labels(filepath, class_names):
    image_list = []
    label_list = []
    for class_n in class_names:  # do dogs and cats
        path = os.path.join(filepath,class_n)  # create path to dogs and cats
        class_num = class_names.index(class_n)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  
            if ".DS_Store" != img:
                filpath = os.path.join(path,img)
#                 print(filpath, class_num)
                image_list.append(filpath)
                label_list.append(class_num)
#     print(image_list, label_list)
    return image_list, label_list


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img)
    img = tf.cast(img, tf.float32)
    return img

def load_image_with_label(image_path, label):
    class_names = ["normal", "defect"]
#     print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.io.decode_bmp(img, channels=IMG_C)
    img = prep_stage(img)
    img = tf.cast(img, tf.float32)
    
    return img, label


def tf_dataset(images_path, batch_size, labels=False, class_names=None):
  

    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def tf_dataset_labels(images_path, batch_size, class_names=None):
    
    filenames, labels = read_data_with_labels(images_path, class_names)
#     print("testing")
#     print(filenames, labels)
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


''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


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


# In[ ]:


def plot_epoch_result(epochs, loss, name):
    plt.plot(epochs, loss, 'g', label=name)
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[ ]:


def conv_block(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(1,1), padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (1, 1), strides=2, padding="same")(input)
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
    # print("testing")


    """ Encoder using resnet50"""
    # for layer in resnet50.layers:
#         resnet50.summary()
    #   print(layer.name)
    s1 = resnet50.get_layer("input_1").output           ## (128 x 128)
    # print(s1)
    s2 = resnet50.get_layer("conv1_relu").output        ## (64 x 64)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (32 x 32)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (16 x 16)
    s5 = resnet50.get_layer("conv4_block6_out").output  ## (8 x 8)

    """ Bridge """
    b1 = resnet50.get_layer("conv5_block3_out").output  ## (4 x 4)

    # print("test")
    # print(b1.get_weights())
    """ Decoder unet"""
    d1 = decoder_block(b1, s5, 128)                     ## (16 x 16)
    d2 = decoder_block(d1, s4, 64)                     ## (32 x 32)
    d3 = decoder_block(d2, s3, 32)                     ## (64 x 64)
    d4 = decoder_block(d3, s2, 16)                      ## (128 x 128)
    d5 = decoder_block(d4, s1, 8)                      ## (128 x 128)

    """ Output """
#     outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d5)
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same")(d5)

    model = tf.keras.models.Model(inputs, outputs)

    return model


# In[ ]:


# create discriminator model
def build_discriminator(inputs):

    x = tf.keras.layers.SeparableConvolution2D(128,kernel_size= (1, 1), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.SeparableConvolution2D(256,kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.SeparableConvolution2D(512,kernel_size= (1, 1), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.SeparableConvolution2D(1024,kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
#     output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    output = tf.keras.layers.Dense(1)(x)
    
    
    model = tf.keras.models.Model(inputs, output)
    return model
    # return x


# In[ ]:


class ResUnetGAN(tf.keras.models.Model):
    def __init__(self, discriminator, generator):
        super(ResUnetGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        # Regularization Rate for each loss function
        self.ADV_REG_RATE_LF = 1
        self.REC_REG_RATE_LF = 50
        self.SSIM_REG_RATE_LF = 50
        self.FEAT_REG_RATE_LF = 1
        self.filelogs = None

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
        
    

    

    def compile(self, d_optimizer, g_optimizer, filepath):
        super(ResUnetGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
        # columns name (epoch, gen_loss, disc_loss)
#         field_names = ['epoch', 'gen_loss', 'disc_loss']
#         logs = pd.DataFrame([], columns=field_names)
#         fileExist = os.path.exists(filepath)
#         if not fileExist:
#             print("file not found. then we create new file")
#             logs.to_csv(filepath, encoding='utf-8', index=False)
        
#         with open(filepath, 'a') as f_object:
#             # Pass this file object to csv.writer()
#             # and get a writer object
#             self.filelogs = DictWriter(f_object, fieldnames=field_names)

            



  
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):


        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print("Images: ", images)
            reconstructed_images = self.generator(images, training=True)
            real_output = self.discriminator(images, training=True)
            # print(generated_images.shape)
            fake_output = self.discriminator(reconstructed_images, training=True)

            # Loss 1: ADVERSARIAL loss
            real_loss = cross_entropy(real_output, tf.ones_like(real_output))
            fake_loss = cross_entropy(fake_output, tf.zeros_like(fake_output))
            disc_adv_loss = real_loss + fake_loss
            
            gen_adv_loss = cross_entropy(fake_output, tf.ones_like(fake_output))
            
            # Loss 2: RECONSTRUCTION loss (L1)
            loss_rec = tf.reduce_mean(mae(images, reconstructed_images))
        
            # Loss 3: SSIM Loss
            loss_ssim =  tf.reduce_mean(ssim(images, reconstructed_images)) 
        
            # Loss 4: FEATURE Loss
            loss_feat = tf.reduce_mean(mse(real_output, fake_output))

            gen_loss = tf.reduce_mean( (gen_adv_loss * self.ADV_REG_RATE_LF) + (loss_rec * self.REC_REG_RATE_LF) + (loss_ssim * self.SSIM_REG_RATE_LF) + (loss_feat * self.FEAT_REG_RATE_LF) )
            disc_loss = tf.reduce_mean( (disc_adv_loss * self.ADV_REG_RATE_LF) + (loss_feat * self.FEAT_REG_RATE_LF) )

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        

        
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        


        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "disc_adv_loss": disc_adv_loss,
            "gen_adv_loss": gen_adv_loss,
            "loss_rec": loss_rec,
            "loss_ssim": loss_ssim,
            "loss_feat": loss_feat
        }

    def saved_model(self, gmodelpath, dmodelpath, num_of_epoch):
        self.generator.save(gmodelpath)
        self.discriminator.save(dmodelpath)

    def loaded_model(self, g_filepath, d_filepath):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        
    # load and save data of training process
    def load_save_processing(self,filepath, epoch, disc_loss, gen_loss, d_filepath, g_filepath, resume=False):
        # columns name (epoch, gen_loss, disc_loss)

        if resume:
            # load logs data
            logs = pd.read_csv(filepath)
            logs.sort_values("epoch", ascending=True)
            epoch = logs['epoch'].tolist()
            disc_loss = logs['disc_loss'].tolist()
            gen_loss = logs['gen_loss'].tolist()
            
            # load model data
            self.loaded_model(g_filepath, d_filepath)
            return epoch, disc_loss, gen_loss
        
        else:
            data={'epoch':epoch,'disc_loss':'disc_loss','gen_loss':gen_loss}
            print("row added." , data)
            self.filelogs.writerow(data)
            return None, None, None
            
            
    def testing(self, filepath, g_filepath, d_filepath, nameModel):
        threshold = 0.8
        class_names = ["normal", "defect"]
        test_dateset = load_image_test(filepath, class_names)
        # print(test_dateset)
        
        # range between 0-1
        anomaly_weight = 0.1
       
       
#         predictions = np.array([])
#         labels =  np.array([])
        scores_ano = []
        real_label = []
        i = 0
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        
        
        for images, labels in test_dateset:
            i += 1
            
            reconstructed_images = self.generator(images, training=False)
            real_output = self.discriminator(images, training=False)
            # print(generated_images.shape)
            fake_output = self.discriminator(reconstructed_images, training=False)

            
            # Loss 2: RECONSTRUCTION loss (L1)
            loss_rec = tf.reduce_mean(mae(images, reconstructed_images))
        
        
#         loss_feat = tf.reduce_mean( tf.keras.losses.mse(real, fake) )
            loss_feat = tf.reduce_mean(mse(real_output, fake_output))

            
            score = (anomaly_weight * loss_rec) + ((1-anomaly_weight) * loss_feat)
#             print(score, loss_rec, loss_feat)
            print(i, score.numpy(),labels.numpy()[0] )
#          
            scores_ano = np.append(scores_ano, score.numpy())
            real_label = np.append(real_label, labels.numpy()[0])
            
        
        
        ''' Scale scores vector between [0, 1]'''
        # scores_ano = (scores_ano - scores_ano.min())/(scores_ano.max()-scores_ano.min())
        # scores_ano = tft.scale_to_0_1(scores_ano)
        scores_ano = (scores_ano - np.min(scores_ano))/np.ptp(scores_ano)
        print("before conversion: ",scores_ano)
        scores_ano = (scores_ano > threshold).astype(int)
        print("scores_ano: ", scores_ano)
        print("real_label: ", real_label)
        auc_out = roc(real_label, scores_ano)
        print("auc: ", auc_out)

        cm = tf.math.confusion_matrix(labels=real_label, predictions=scores_ano).numpy()
        print(cm)
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[0][0]
        plot_confusion_matrix(cm, class_names, title=nameModel)


        diagonal_sum = cm.trace()
        sum_of_all_elements = cm.sum()

        print("Accuracy: ", diagonal_sum / sum_of_all_elements )
        print("False Alarm Rate: ", FP/(FP+TP))
        print("Leakage Rate: ", FN/(FN+TN))
#         print("precision_score: ",precision_score(real_label, scores_ano))
#         print("recall_score: ",recall_score(real_label, scores_ano))
        
        
    def checking_gen_disc(self, g_filepath, d_filepath):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)
        path = "mura_data/mura_data/train_data/normal/normal_7A1D30N6KAZZ_20210401041347_0_L050P_resize.bmp.bmp"
        image = tf.keras.preprocessing.image.load_img(path, target_size=(128,128))
        
        array_image = tf.keras.preprocessing.image.img_to_array(image)
        
        array_image = tf.reshape(array_image, (-1, 128, 128, 3))
        images = tf_clahe.clahe(array_image)
        plt.figure()
        plt.imshow(images)
        reconstructed_images = self.generator(images, training=False)
        
        reconstructed_images = tf.cast(reconstructed_images[0], tf.float32)
        plt.figure()
        plt.imshow(reconstructed_images)


# In[ ]:


if __name__ == "__main__":
    # run the function here
    nameModel= "normal_clahe"
    print("start: ", nameModel)
    ## Hyperparameters
    batch_size = 25
    num_epochs = 2
    
    train_images_path = "mura_data/playground/train_data/*.bmp"
    test_data_path = "mura_data/playground/test_data"
    saved_model_path = "mura_data/mura_data/saved_model/"
    logsPath = "mura_data/mura_data/logs/"

    
    pathGmodal = saved_model_path + nameModel + "g_model" + str(num_epochs) + ".h5"
    pathDmodal = saved_model_path +  nameModel + "d_model" + str(num_epochs) + ".h5"
    pathLogs = logsPath + "logs_" + nameModel + str(num_epochs) + ".csv"
    
    input_shape = (IMG_W, IMG_H, IMG_C)
    # print(input_shape)

    """ Input """
    inputs = tf.keras.layers.Input(input_shape, name="input_1")

    

    
    d_model = build_discriminator(inputs)
    
    g_model = build_generator_resnet50_unet(inputs)
#     print("done")
#     d_model.summary()
#     g_model.summary()
    
    resunetgan = ResUnetGAN(d_model, g_model)


    g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
    resunetgan.compile(d_optimizer, g_optimizer, pathLogs)

    # print(train_images_path)
    train_images = glob(train_images_path)
    train_images_dataset = load_image_train(train_images, batch_size)

    array_elapsed = []
    epochs_list = []
    gen_loss_list = []
    disc_loss_list = []
    skip_epoch = 0
    
    
    for epoch in range(num_epochs+1):
#         if epoch <= skip_epoch:
#             continue
        print("Epoch: ", epoch)
        now = datetime.now()
#         for image_batch in train_images_dataset:
#             print(image_batch.shape)
#             print("Images_batch: ", image_batch)
        r = resunetgan.fit(train_images_dataset)
        epochs_list.append(epoch)
        gen_loss_list.append(r.history["gen_loss"][0])
        disc_loss_list.append(r.history["disc_loss"][0])
#         print(r.history["gen_loss"][0], r.history["disc_loss"][0] )
        if (epoch + 1) % 15 == 0 or (epoch + 1) <= 15:
            resunetgan.saved_model(pathGmodal, pathDmodal, num_epochs)
            print('saved for epoch',epoch + 1)
        
                   
        later = datetime.now()
        elapsed_time =  (later - now).total_seconds()
        array_elapsed = np.append(array_elapsed, elapsed_time)
        print("Time Consumend of this epoch: ", elapsed_time)
    
    print("Duration of trainning Data: ", np.sum(array_elapsed), " seconds")
#     print(epochs_list)
#     print(gen_loss_list)
#     print(disc_loss_list)
    plot_epoch_result(epochs_list, gen_loss_list, "Generator Loss")
    plot_epoch_result(epochs_list, disc_loss_list, "Discriminator Loss")
    resunetgan.testing(test_data_path, pathGmodal, pathDmodal, nameModel)
    
    
#     resunetgan.checking_gen_disc(pathGmodal, pathDmodal)


# In[ ]:




