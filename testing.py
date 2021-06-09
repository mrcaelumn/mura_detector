import tensorflow as tf
import matplotlib.pyplot as plt

img = tf.io.read_file("mura_data/grayscale/defect.bmp")
img = tf.io.decode_bmp(img)

plt.imshow(img)