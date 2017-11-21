import tensorflow as tf


n_classes = 2
img_dim = 100
n_channels = 3

x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])