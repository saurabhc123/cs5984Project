import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy as sc
import ConvHelper
import pickle
import csv as csv
import datetime
import os as os

current_working_folder = os.path.dirname(os.getcwd())

metadata_files_path = os.path.join(current_working_folder, 'Project/Datasets/Adience/')
train_metadata_filename = 'downsampled_gender_train.txt'
test_metadata_filename = 'downsampled_gender_train.txt'
image_files_path = os.path.join(current_working_folder, 'Project/Datasets/Adience/aligned/')
STEPS = 50
MINIBATCH_SIZE = 100
n_classes = 2
img_dim = 100
n_channels = 3

class AdienceLoader(object):
    def __init__(self, source_files, labels):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = labels

    def load(self):
        data = self._source
        imgs = [image_files_path + d for d in self._source]
        images = np.asarray([plt.imread(fname, format='jpeg') for fname in imgs])
        #im = plt.imread(imgs[0], format='jpeg')
        #plt.imshow(im)
        #plt.show()
        #display_image(images[0:4],2)
        n = len(images)
        self.images = images.reshape(n, n_channels, img_dim, img_dim).transpose(0, 2, 3, 1)\
                          .astype(float) / 255
        self.labels = one_hot(np.hstack([d for d in self.labels]), n_classes)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size],self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

class AdienceDataManager(object):
    def __init__(self):
        train_data_image_filenames , train_labels = load_data(metadata_files_path + train_metadata_filename)
        #imgs = [plt.imread(fname)[...,:n_channels] for fname in train_data_image_filenames]
        self.train = AdienceLoader(train_data_image_filenames, train_labels).load()
        test_data_image_filenames , test_labels = load_data(metadata_files_path + test_metadata_filename)
        self.test = AdienceLoader(test_data_image_filenames, test_labels).load()


def one_hot(vec, vals=n_classes):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def load_data(inputFilenameWithPath):
    data = []
    labels = []
    with open(inputFilenameWithPath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            data.append(row[0])
            labels.append(int(row[1]))
    #print(len(data))
    return data[0:12000] , labels[0:12000]

def display_image(images, size):
    n = len(images)
    size = n/2
    plt.figure()
    plt.gca().set_axis_off()
    p = images[np.random.choice(n)]
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    #im = images.reshape(1,img_dim,img_dim,n_channels)
    plt.imshow(im)
    plt.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape,)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

def test(sess):
    print ("Starting Test")
    number_of_test_batches = 10
    number_of_samples_per_batch = len(adience.test.images)/number_of_test_batches
    #print len(adience.test.images)
    X = adience.test.images.reshape(number_of_test_batches, number_of_samples_per_batch, img_dim, img_dim, n_channels)
    Y = adience.test.labels.reshape(number_of_test_batches, number_of_samples_per_batch, n_classes)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i],
                                                 keep_prob: 1.0})
                   for i in range(number_of_test_batches)])
    print ("Accuracy: {:.4}%".format(acc * 100))
adience = AdienceDataManager()
#images = adience.train.next_batch(1)
#display_image(images, 2)

x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

conv1 = ConvHelper.conv_layer(x, shape=[7, 7, 3, 32])
conv1_pool = ConvHelper.max_pool_2x2(conv1)
print (conv1_pool)
conv2 = ConvHelper.conv_layer(conv1_pool, shape=[5, 5, 32, 16])
conv2_pool = ConvHelper.max_pool_2x2(conv2)
print (conv2_pool)
conv3 = ConvHelper.conv_layer(conv2_pool, shape=[3, 3, 16, 8])
conv3_pool = ConvHelper.max_pool_2x2(conv3)
print (conv3_pool)
conv3_flat = tf.reshape(conv3_pool, [-1, 13 * 13 * 8])
#conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.elu(ConvHelper.full_layer(conv3_flat, 512))
full_2 = tf.nn.elu(ConvHelper.full_layer(full_1, 512))
#full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = ConvHelper.full_layer(full_2, n_classes)



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_conv,
                                                               labels=y_))
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


STEPS = 70
MINIBATCH_SIZE = 20

print "Starting at:" , datetime.datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print "Initialization done at:" , datetime.datetime.now()
    for epoch in range(STEPS):
        print "Starting epoch", epoch, " at:", datetime.datetime.now()
        for batch_count in range(5):
            batch = adience.train.next_batch(MINIBATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        if(epoch%1 == 0):
            test(sess)

    test(sess)