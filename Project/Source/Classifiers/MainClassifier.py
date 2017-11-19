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
import random

model_folder_name = "models/adience"
model_filename = os.path.join(model_folder_name,"adience_model.ckpt")
STEPS = 50
MINIBATCH_SIZE = 100
n_classes = 2
img_dim = 100
n_channels = 3




x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)


def get_fc7_representation(train):
    i = 10



def train(sess,train,retrain):
    full_2 = get_fc7_representation(train)
    y_conv = ConvHelper.full_layer(full_2, n_classes)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_conv,
                                                                   labels=y_))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    STEPS = 10
    MINIBATCH_SIZE = 20

    if os.path.exists(model_folder_name) & (not retrain):
        print("Model found in file: %s" % model_filename)
        saver.restore(sess, model_filename)
    else:
        if (retrain) & os.path.exists(model_folder_name):
            print ("Retraining the model.")
        print "Starting at:" , datetime.datetime.now()
        sess.run(tf.global_variables_initializer())
        print "Initialization done at:" , datetime.datetime.now()
        for epoch in range(STEPS):
            print "Starting epoch", epoch, " at:", datetime.datetime.now()
            for batch_count in range(len(adience.train.images)/MINIBATCH_SIZE):
                batch = adience.train.next_batch(MINIBATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
            if(epoch%1 == 0):
                test(sess,accuracy)
        save_path = saver.save(sess, model_filename)
        print("Model saved in file: %s" % save_path)
    return accuracy

