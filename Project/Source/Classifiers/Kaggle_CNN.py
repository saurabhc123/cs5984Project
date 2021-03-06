import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
import scipy as sc
from sklearn.model_selection import train_test_split
import ConvHelper
import pickle
import csv as csv
import datetime
import os as os
import random
import Placeholders

import LSTMClassifier
import GRUClassifier
import ImageClassifier

current_working_folder = os.path.dirname(os.getcwd())

# metadata_files_path = os.path.join(current_working_folder, 'Project/Datasets/Adience/')
# train_metadata_filename = 'downsampled48_gender_train.txt'
# test_metadata_filename = 'downsampled48_gender_train.txt'
#image_files_path = os.path.join(current_working_folder, 'Project/Datasets/Adience/aligned/')
model_folder_name = "models/adience"
model_filename = os.path.join(model_folder_name,"adience_model.ckpt")


metadata_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
image_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/ProfileImages')
train_metadata_filename = 'Kaggle-str-process.csv'
serialized_train_metadata_filename = 'KaggleTwitter.json'
test_metadata_filename = 'Kaggle-str-process.csv'

STEPS = 50
MINIBATCH_SIZE = 100
n_channels = 3

class KaggleLoader(object):
    def __init__(self, source_files, labels):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = labels

    def load(self):
        data = self._source
        imgs = [image_files_path + '/' + d + '.jpg' for d in self._source]
        images = np.asarray([plt.imread(fname, format='jpg') for fname in imgs])
        n = len(images)
        self.images = images.reshape(n, n_channels, Placeholders.img_dim, Placeholders.img_dim).transpose(0, 2, 3, 1)\
                          .astype(float) / 255
        self.labels = one_hot(np.hstack([d for d in self.labels]), Placeholders.n_classes)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size],self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

class KaggleImageDataManager(object):
    def __init__(self):
        self.data = []
        train_data_image_filenames , train_labels = self.load_data(metadata_files_path + train_metadata_filename)
        train_d, test_d, train_labels, test_labels = train_test_split(train_data_image_filenames, train_labels, test_size=0.3, random_state= random.randint(0,100) )
        test_d, validation_d, test_labels, validation_labels = train_test_split(test_d, test_labels, test_size=0.5, random_state = random.randint(0,100) )
        self.train = KaggleLoader(train_d, train_labels).load()
        print("Training data size:" + str(len(train_labels)))
        self.test = KaggleLoader(test_d, test_labels).load()
        self.validation = KaggleLoader(validation_d, validation_labels).load()
        print("Test data size:" + str(len(test_labels)))

    def load_data(self, inputFilenameWithPath):
        data = []
        labels = []
        with open(inputFilenameWithPath, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip header row
            for row in reader:
                try:
                    data.append(self.get_image_data(row[5], row[4]))
                    labels.append(int(0) if row[0] == 'male' else int(1))
                except Exception as ex:
                    print(ex)
                    print("Omitting:", row[4])
                    pass

        return data , labels

    def get_image_data(self, image_file, name):
        file_name = image_file.split('/')[-1]
        file_extension = file_name.split('.')[-1]
        full_file_name = os.path.join(image_files_path, name + '.' + file_extension)
        image_data = plt.imread(full_file_name, format=file_extension)
        return image_data

    def get_clean_kaggle_dataset(self):
        goodData = []
        for sample in self.data:
            try:
                img_data = sample.get_image_data()
                goodData.append(sample)
            except Exception as ex:
                print(ex)
                pass
                #print("Omitting:", sample.name)
        return goodData


def one_hot(vec, vals=Placeholders.n_classes):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out



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

def validate(sess, accuracy):
    print ("Starting Test")
    number_of_test_batches = 10
    number_of_samples_per_batch = 20
    total_samples = number_of_test_batches * number_of_samples_per_batch
    random_index = random.randint(0,len(adience.validation.images) - total_samples)

    #print len(adience.test.images)
    X = adience.validation.images
    Y = adience.validation.labels
    perform_evaluation(X, Y, accuracy, sess, "Validation")


def perform_evaluation(X, Y, accuracy, sess, test_type, loss = None):
    number_of_test_batches = 10
    number_of_samples_per_batch = int(len(X)/number_of_test_batches)
    total_samples = number_of_test_batches * number_of_samples_per_batch

    x_total = X[:total_samples]
    y_total = Y[:total_samples]

    X = x_total.reshape(number_of_test_batches, number_of_samples_per_batch, Placeholders.img_dim, Placeholders.img_dim, n_channels)
    Y = y_total.reshape(number_of_test_batches, number_of_samples_per_batch, Placeholders.n_classes)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i],
                                                 keep_prob: 1.0})
                   for i in range(number_of_test_batches)])
    if loss is not None:
        mse = np.sum([loss.eval(feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                 for i in range(number_of_test_batches)])
        print(test_type + "Loss: {:.4}".format(mse))
    print(test_type + "Accuracy: {:.4}%".format(acc * 100))


def test_on_train(sess, accuracy, loss):
    print ("Starting Test")
    number_of_test_batches = 10
    number_of_samples_per_batch = 20
    total_samples = number_of_test_batches * number_of_samples_per_batch
    random_index = random.randint(0,len(adience.test.images) - total_samples)

    #print len(adience.test.images)
    X = adience.train.images
    Y = adience.train.labels
    perform_evaluation(X, Y, accuracy, sess, "Training",loss)

def test(sess, accuracy):
    print ("Starting Test")
    number_of_test_batches = 10
    number_of_samples_per_batch = 20
    total_samples = number_of_test_batches * number_of_samples_per_batch
    random_index = random.randint(0,len(adience.test.images) - total_samples)

    #print len(adience.test.images)
    X = adience.test.images
    Y = adience.test.labels
    perform_evaluation(X, Y, accuracy, sess, "Test")

def get_fc7_representation(sample, sess, fc7):
    image = np.array(sample).reshape((1, Placeholders.img_dim, Placeholders.img_dim, n_channels))
    print (image.shape)
    fc7rep = sess.run(fc7, feed_dict= {x : image , keep_prob: 1.0})
    return fc7rep

def train(sess, adience, retrain = False):
    conv1 = ConvHelper.conv_layer(x, shape=[7, 7, 3, 96], strides = [1, 2, 2, 1])
    conv1_pool = ConvHelper.max_pool_2x2(conv1)
    print (conv1_pool)
    conv2 = ConvHelper.conv_layer(conv1_pool, shape=[5, 5, 96, 256])
    conv2_pool = ConvHelper.max_pool_2x2(conv2)
    print (conv2_pool)
    conv3 = ConvHelper.conv_layer(conv2_pool, shape=[3, 3, 256, 384])
    conv3_pool = ConvHelper.max_pool_2x2(conv3)
    print (conv3_pool)
    conv3_flat = tf.reshape(conv3_pool, [-1, 3 * 3 * 384])

    with tf.variable_scope("FC-7"):
        fully_connected1_dropout = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)
        full_1 = tf.nn.relu(ConvHelper.full_layer(fully_connected1_dropout, Placeholders.img_feature_width))
        fully_connected2_dropout = tf.nn.dropout(full_1, keep_prob=keep_prob)
        fc7layer = tf.nn.relu(ConvHelper.full_layer(fully_connected2_dropout, Placeholders.img_feature_width))


    y_conv = ConvHelper.full_layer(fc7layer, Placeholders.n_classes)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_conv,
                                                                   labels=y_))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    STEPS = 3
    MINIBATCH_SIZE = 20

    if os.path.exists(model_folder_name) & (not retrain):
        print("Model found in file: %s" % model_filename)
        saver.restore(sess, model_filename)
    else:
        if (retrain) & os.path.exists(model_folder_name):
            print ("Retraining the model.")
        print ("Starting at:" , datetime.datetime.now())
        sess.run(tf.global_variables_initializer())
        print ("Initialization done at:" , datetime.datetime.now())
        for epoch in range(STEPS):
            print ("Starting epoch", epoch, " at:", datetime.datetime.now())
            for batch_count in range(int(len(adience.train.images)/MINIBATCH_SIZE)):
                batch = adience.train.next_batch(MINIBATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob: 0.75})
            if(epoch%10 == 0):
                test_on_train(sess, accuracy, loss)
                validate(sess, accuracy)
        test(sess, accuracy)
        save_path = saver.save(sess, model_filename)
        print("Model saved in file: %s" % save_path)
    return accuracy, fc7layer

adience = KaggleImageDataManager()
x = Placeholders.x
y_ = Placeholders.y_

keep_prob = Placeholders.adience_keep_prob
with tf.Session() as sess:
    accuracy, fc7 = train(sess, adience, retrain=True)
    image = np.array(adience.train.next_batch(1)[0]).reshape((1,Placeholders.img_dim, Placeholders.img_dim,n_channels))
    print (image.shape)
    fc7rep = get_fc7_representation(image, sess, fc7)
    print (fc7rep.shape)
    validate(sess, accuracy)
    #with tf.variable_scope("main_classifier"):
        #ImageClassifier.train(sess, None, True, fc7)

