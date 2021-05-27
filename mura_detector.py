# -*- coding: utf-8 -*-
"""mura_detector.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xwg6B8PUF5SqSHRPDdsWOTX-W5s8c7Kd
"""

# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.initializers import RandomNormal
# from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import tensorflow_datasets as tfds
from glob import glob
import time

IMG_H = 128
IMG_W = 128
IMG_C = 3  ## Change this to 1 for grayscale.


# Regularization Rate for each loss function
ADV_REG_RATE_LF = 1
REC_REG_RATE_LF = 50
SSIM_REG_RATE_LF = 50
FEAT_REG_RATE_LF = 1

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss function for evaluating adversarial loss
adv_loss_fn = tf.losses.MeanSquaredError()

w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.io.decode_bmp(img)
  img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
  img = tf.cast(img, tf.float32)
  img = (img - 127.5) / 127.5
  return img

def tf_dataset(images_path, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(images_path)
  dataset = dataset.shuffle(buffer_size=10240)
  dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

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

# create generator model based on resnet50 and unet network
def build_generator_resnet50_unet(input_shape):
    
    # print(inputs)
    # print("pretained start")
    """ Pre-trained ResNet50 Model """
    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_shape)
    # print("testing")
    """ Encoder using resnet50"""
    # for layer in resnet50.layers:
    #   print(layer.name)
    s1 = resnet50.get_layer("input_1").output           ## (128 x 128)
    # print(s1)
    s2 = resnet50.get_layer("conv1_relu").output        ## (64 x 64)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (32 x 32)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (16 x 16)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)
    
    """ Decoder unet"""
    d1 = decoder_block(b1, s4, 128)                     ## (16 x 16)
    d2 = decoder_block(d1, s3, 64)                     ## (32 x 32)
    d3 = decoder_block(d2, s2, 32)                     ## (64 x 64)
    d4 = decoder_block(d3, s1, 16)                      ## (128 x 128)

    """ Output """
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# create discriminator model
def build_discriminator():
  # Load the pre-trained model and freeze it.
  pre_trained = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, IMG_C)
    )
  pre_trained.trainable = False  # mark all weights as non-trainable
  # Define a Sequential model, adding trainable layers on top of the previous.
  
  model = tf.keras.Sequential([pre_trained])

  model.add(tf.keras.layers.SeparableConvolution2D(32,kernel_size= (1, 1), strides=(2, 2), padding='same'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.SeparableConvolution2D(64,kernel_size=(1, 1), strides=(2, 2), padding='same'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1))
  
  return model

# loss function for SSIM
def SSIMLoss(x_actual, x_recon):
  return 1 - tf.reduce_mean(tf.image.ssim(x_actual, x_recon, 1.0))

# loss functio for adversial
def ADVLoss(x_actual, x_recon):
  loss_act = tf.losses.MeanSquaredError(tf.ones_like(x_actual), x_actual)
  loss_rec = tf.losses.MeanSquaredError(tf.zeros_like(x_recon), x_recon)
  return tf.losses.me(x_actual, x_recon)

# loss function for feature
def FEATLoss(x_actual, x_recon):
  return tf.keras.losses.mean_absolute_error(x_actual, x_recon)

# loss function for reconstruction
def RECONLoss(x_actual, x_recon):
  print("recon")
  return tf.keras.losses.mae(x_recon, x_actual)

def gen_loss_func(x_actual, x_recon):
  #test
  # print("custom loss function for generator")
  return (SSIM_REG_RATE_LF * SSIMLoss(x_actual, x_recon)) + (ADV_REG_RATE_LF * ADVLoss(x_actual, x_recon)) + (FEAT_REG_RATE_LF * FEATLoss(x_actual, x_recon)) + (REC_REG_RATE_LF * RECONLoss(x_recon, x_actual))


def generator_loss(x_recon):
  # return cross_entropy(tf.ones_like(fake_output), fake_output)
  recon_loss = adv_loss_fn(tf.ones_like(x_recon), x_recon)
  # return recon_loss * ADV_REG_RATE_LF
  return recon_loss

def disc_loss_func(x_pred, x_tar):
  #test
  # print("custom loss function for discrimnator")
  return (ADV_REG_RATE_LF * ADVLoss(x_actual, x_recon)) + (FEAT_REG_RATE_LF * FEATLoss(x_actual, x_recon))

def discriminator_loss(x_actual, x_recon):
  act_loss = adv_loss_fn(tf.ones_like(x_actual), x_actual)
  recon_loss = adv_loss_fn(tf.zeros_like(x_recon), x_recon)
  # return (act_loss + recon_loss) * ADV_REG_RATE_LF
  return (act_loss + recon_loss) * 0.5

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

class ResUnetGAN(tf.keras.models.Model):
  def __init__(self, discriminator, generator, latent_dim, batch_size):
    super(ResUnetGAN, self).__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.batch_size = batch_size

  def compile(self, d_optimizer, g_optimizer, gen_loss_fn, disc_loss_fn):
    super(ResUnetGAN, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.gen_loss_fn = gen_loss_fn
    self.disc_loss_fn = disc_loss_fn
  
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images):
    batch_size = tf.shape(images)[0]
    # print(batch_size, IMG_W, IMG_H, IMG_C)
    noise = tf.random.normal(shape=(batch_size, IMG_W, IMG_H, IMG_C))
    abnormal_scores = []

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(images, training=True) 
      real_output = self.discriminator(images, training=True)  
      # print(generated_images.shape)
      fake_output = self.discriminator(generated_images, training=True)  
      gen_loss = self.gen_loss_fn(fake_output)
      disc_loss = self.disc_loss_fn(real_output, fake_output) 
    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
  
    return {"gen_loss": gen_loss, "disc_loss": disc_loss}

if __name__ == "__main__":
  # run the function here
  print("start")
  ## Hyperparameters
  batch_size = 24
  latent_dim = 128
  input_shape = (IMG_W, IMG_H, IMG_C)
  # print(input_shape)

  """ Input """
  inputs = tf.keras.layers.Input(input_shape, name="input_1")
 
  num_epochs = 1
  train_images_path = glob("/content/drive/MyDrive/mura_data/train_data/*")
  test_images_path = glob("/content/drive/MyDrive/mura_data/test_data/*.bmp")

  d_model = build_discriminator()
  g_model = build_generator_resnet50_unet(inputs)

  d_model.load_weights("saved_model/d_model_500.h5")
  g_model.load_weights("saved_model/g_model_500.h5")

  # d_model.summary()
  # g_model.summary()

  resunetgan = ResUnetGAN(d_model, g_model, latent_dim, batch_size)



  g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
  d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
  resunetgan.compile(d_optimizer, g_optimizer, generator_loss, discriminator_loss)
  
  # print(resunetgan) 

  # print(train_images_path)
  train_images_dataset = tf_dataset(train_images_path, batch_size)
  test_images_dataset = tf_dataset(test_images_path, batch_size)

  # resunetgan.fit(train_images_dataset)
  
  
  for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    start = time.time()
    for image_batch in train_images_dataset:
      # print(image_batch.shape)
      resunetgan.fit(image_batch)

    # for test_batch in test_images_dataset:
    #   result = resunetgan.evaluate(test_batch)
    #   print(result)
  #     g_model.save("saved_model/g_model.h5")
  #     d_model.save("saved_model/d_model.h5")

  # resunetgan.summary()
  # resunetgan.save_weights("saved_model/resunet_model")

import numpy as np
# load an image
def load_image_test(filename, size=(128,128)):
	# load image with the preferred size
	pixels = tf.keras.preprocessing.image.load_img(filename, target_size=size)
	# convert to numpy array
	pixels = tf.keras.preprocessing.image.img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = np.expand_dims(pixels, 0)
	return pixels

# test_images_dataset = tf_dataset(test_images_path, batch_size)
normal_images = glob('/content/drive/MyDrive/mura_data/test_data/normal_*.bmp')
defect_images = glob('/content/drive/MyDrive/mura_data/test_data/defect_*.bmp')
len_nor_data = len(normal_images)
len_def_data = len(defect_images)
print(len_nor_data)
print(len_def_data)
threshold = 0.6
defect_preds = []
for image in defect_images:
  # print(image)
  if "DS_Store" not in image:
    src_image = load_image_test(image)

    test = d_model.predict(src_image)
    test = (test + 1) / 2.0
    defect_preds = np.append(defect_preds,test)

    # preds = (preds - preds.min())/(preds.max()-preds.min())
    # print(test)



normal_preds = []
for image in normal_images:
  # print(image)
  if "DS_Store" not in image:
    src_image = load_image_test(image)

    test = d_model.predict(src_image)
    test = (test + 1) / 2.0
    normal_preds = np.append(normal_preds,test)

    # preds = (preds - preds.min())/(preds.max()-preds.min())
    # print(test)


print(defect_preds)
print(np.mean(defect_preds))
true_def_pred = len(np.where(defect_preds > threshold)[0])
print(true_def_pred)


print(normal_preds)
print(np.mean(normal_preds))
true_nor_pred = len(np.where(normal_preds < threshold)[0])
print(true_nor_pred)

total_acc = (true_def_pred + true_nor_pred) / (len_nor_data + len_def_data) * 100
print("total_accuracy: ", total_acc)

