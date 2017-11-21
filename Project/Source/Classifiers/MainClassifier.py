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
import Placeholders

model_folder_name = "models/adience"
model_filename = os.path.join(model_folder_name,"main_model.ckpt")
STEPS = 50
MINIBATCH_SIZE = 100
n_classes = 2
feature_width = 512
img_dim = 100
n_channels = 3



#x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
x_main = tf.placeholder(tf.float32, shape=[None, feature_width])
#y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)


def get_fc7_representation(sample, sess, fc7, x):
    image = np.array(sample).reshape((-1,img_dim,img_dim,n_channels))
    #print image.shape
    fc7rep = sess.run(fc7, feed_dict= {x : image})
    return fc7rep


x = Placeholders.x
y_ = Placeholders.y_


def train(sess, train, retrain, fc7, adience):

    fully_connected = tf.nn.elu(ConvHelper.full_layer(x_main , 512))
    y_conv = ConvHelper.full_layer(fully_connected, n_classes)

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

    #Retrieve training data

    #training_data = get_fc7_representation(train, sess, fc7)

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
                #fc7_x = map(lambda image: get_fc7_representation(image, sess, fc7), batch[0])
                fc7_x = get_fc7_representation(batch[0], sess, fc7,x)
                sess.run(train_step, feed_dict={x_main: fc7_x, y_: batch[1],keep_prob: 1.0})
            if(epoch%1 == 0):
                acc = sess.run(accuracy, feed_dict={x_main: fc7_x, y_: batch[1],
                                                 keep_prob: 1.0})
                print ("Accuracy: {:.4}%".format(acc * 100))
                #test(sess,accuracy, adience)
        save_path = saver.save(sess, model_filename)
        print("Model saved in file: %s" % save_path)
    return accuracy

