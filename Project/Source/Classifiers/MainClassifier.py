import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
import scipy as sc
import ConvHelper
import pickle
import csv as csv
import datetime
import os as os
import random
import Placeholders
import KaggleDataManager
import word2Vec as w2v
import webcolors as wc

model_folder_name = "models/adience"
model_filename = os.path.join(model_folder_name,"main_model.ckpt")
STEPS = 50
MINIBATCH_SIZE = 100
n_classes = 2
img_feature_width = 512
word_vec_length = 600
profile_color_feature_length = 6
feature_width = img_feature_width + word_vec_length + profile_color_feature_length
img_dim = 100
n_channels = 3

#Kaggle data
current_working_folder = os.path.dirname(os.getcwd())
kaggle_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
kaggle_images_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/Images')
train_metadata_filename = 'Kaggle-str-process.csv'
test_metadata_filename = 'Kaggle-str-process.csv'
today = datetime.datetime.now()
format = "%d_%m_%Y_%H_%M_%S"
run_folder = today.strftime(format)


x_main = tf.placeholder(tf.float32, shape=[None, feature_width])
keep_prob = tf.placeholder(tf.float32)


def get_fc7_representation(sample, sess, fc7):
    image = np.array(sample).reshape((-1,img_dim,img_dim,n_channels))
    fc7rep = sess.run(fc7, feed_dict= {x : image})
    return np.array(fc7rep)


x = Placeholders.x
y_ = Placeholders.y_

def test(sess, accuracy, test,fc7, word_vec, y_conv, correct_prediction,loss, epoch, datasetType = "Test"):
    print ("Starting Test")
    number_of_test_batches = 5
    number_of_samples_per_batch = 10
    total_samples = number_of_test_batches * number_of_samples_per_batch
    random_index = random.randint(0,len(test) - total_samples)

    #print len(adience.test.images)
    data = test[random_index:random_index+total_samples]
    batch = get_features_and_labels(data, sess, fc7, word_vec)
    batch_x = batch[0].reshape(-1, feature_width)

    acc = sess.run(accuracy, feed_dict={x_main: batch_x, y_: batch[1],
                                     keep_prob: 1.0})
    print (datasetType + " Accuracy: {:.4}%".format(acc * 100))
    mse = loss.eval(feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0})
    predictions = np.array(sess.run(tf.argmax(y_conv, 1), feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0}))
    correct_predictions = np.array(sess.run(correct_prediction, feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0}))
    write_results_to_file(mse, acc * 100, data, predictions, correct_predictions, epoch, datasetType)
    return acc * 100


def test_all(sess, accuracy, test,fc7, word_vec, y_conv, correct_prediction,loss, epoch = 0, datasetType = "Test All"):
    print ("Starting Test All")

    #print len(adience.test.images)
    data = test
    batch = get_features_and_labels(data, sess, fc7, word_vec)
    batch_x = batch[0].reshape(-1, feature_width)

    acc = sess.run(accuracy, feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0})
    print (datasetType + " Accuracy: {:.4}%".format(acc * 100))
    mse = loss.eval(feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0})
    predictions = np.array(sess.run(tf.argmax(y_conv, 1), feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0}))
    correct_predictions = np.array(sess.run(correct_prediction, feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0}))
    write_results_to_file(mse, acc * 100, data, predictions, correct_predictions, epoch, datasetType)


def test1(sess, accuracy, test,fc7, word_vec):
    print ("Starting Test")
    batch = get_features_and_labels(test, sess, fc7, word_vec)
    batch_x = batch[0].reshape(-1, feature_width)

    acc = sess.run(accuracy, feed_dict={x_main: batch_x, y_: batch[1],
                                     keep_prob: 1.0})
    print ("Accuracy: {:.4}%".format(acc * 100))

def train(sess, train, retrain, fc7):
    output_folder = os.path.join(current_working_folder,"Project/output")
    output_folder = os.path.join(output_folder, run_folder)
    if not os.path.exists(output_folder):
        print("Creating folder: " + output_folder)
        os.makedirs(output_folder)
    else:
        print("Folder exists: " + output_folder)

    kdm = KaggleDataManager.KaggleDataManager(kaggle_files_path + train_metadata_filename)
    word_vec = w2v.word2vec()

    fully_connected = tf.nn.relu(ConvHelper.full_layer(x_main, feature_width))
    #fully_connected = tf.nn.relu(ConvHelper.full_layer(fully_connected1 , feature_width))
    fully_connected_dropout = tf.nn.dropout(fully_connected, keep_prob=keep_prob)
    y_conv = ConvHelper.full_layer(fully_connected_dropout, n_classes)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_conv,
                                                                   labels=y_))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    STEPS = 500
    MINIBATCH_SIZE = 50

    #Retrieve training data

    #training_data = get_fc7_representation(train, sess, fc7)

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
            for batch_count in range(int(len(kdm.train)/MINIBATCH_SIZE)):
                batch = get_features_and_labels(kdm.next_batch(MINIBATCH_SIZE), sess, fc7, word_vec)
                batch_x = batch[0].reshape(-1, feature_width)
                sess.run(train_step, feed_dict={x_main: batch_x, y_: batch[1],keep_prob: 0.5})
            if(epoch%10 == 0):
                #acc = sess.run(accuracy, feed_dict={x_main: batch_x, y_: batch[1], keep_prob: 1.0})
                #print ("Accuracy: {:.4}%".format(acc * 100))
                mse = loss.eval(feed_dict={x_main: batch_x, y_: batch[1],keep_prob: 0.5})
                print("Iter " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(mse))
                train_accuracy = test(sess, accuracy, kdm.train, fc7, word_vec, y_conv, correct_prediction, loss, epoch, datasetType="Train")
                test_accuracy = test(sess, accuracy, kdm.test, fc7, word_vec, y_conv, correct_prediction, loss, epoch, datasetType="Test")
                if (test_accuracy > Placeholders.best_accuracy_so_far):
                    Placeholders.best_accuracy_so_far = test_accuracy
                    test_all(sess, accuracy, kdm.test, fc7, word_vec, y_conv, correct_prediction, loss, epoch)
                elif (train_accuracy > 80):
                    test_all(sess, accuracy, kdm.test, fc7, word_vec, y_conv, correct_prediction, loss, epoch)
        test_all(sess, accuracy, kdm.test, fc7, word_vec, y_conv, correct_prediction, loss)
        save_path = saver.save(sess, model_filename)
        print("Model saved in file: %s" % save_path)
    return accuracy



def get_features_and_labels(batch_data, sess, fc7, word_vec):
    labels = list(map(lambda x: x.label , batch_data))
    features = list(map(lambda x: get_feature_from_sample(x, sess, fc7, word_vec), batch_data))
    #print(labels.shape , images.shape)
    labels = one_hot(np.hstack([d for d in labels]), n_classes)
    #print(features.shape , labels.shape)
    return np.array(features), np.array(labels)

def get_feature_from_sample(x, sess, fc7, word_vec):
    features = np.array([]).reshape((1,0))
    fc7_x = get_fc7_representation(x.get_image_data(), sess, fc7)
    desc_word_vector = word_vec.get_sentence_vector(x.description)
    tweet_word_vector = word_vec.get_sentence_vector(x.tweet_text)
    sidebar_feature = hex_to_rgb(x.sidebar_color)
    link_color_feature = hex_to_rgb(x.link_color)
    features = np.hstack((features,fc7_x))
    features = np.hstack((features,desc_word_vector))
    features = np.hstack((features,tweet_word_vector))
    features = np.hstack((features,sidebar_feature))
    features = np.hstack((features,link_color_feature))
    features = features.reshape(-1, feature_width)
    return features


def hex_to_rgb(hex_color):
    base_color = np.zeros(shape=[1,3])
    if(hex_color is None):
        return base_color
    try:
        color = np.array(wc.hex_to_rgb("#" + hex_color))
        return color.reshape(1,3)
    except:
        pass
    return base_color


def one_hot(vec, vals=n_classes):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def write_results_to_file(loss, accuracy, test_data_raw, predictions, correct_predictions, epoch, datasetType):
    today = datetime.datetime.now()
    format = "_%d_%m_%Y_%H_%M_%S"
    filename = "output/" + run_folder + "/" +"Result_"+ datasetType + today.strftime(format) + "_Iteration_" + str(epoch)  + "_Accuracy_" + str(round(accuracy, 2)) + ".csv"
    with open(filename, 'wt') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([datasetType + " Accuracy = " + str(accuracy)])
        wr.writerow(["Loss = " + str(loss)])
        wr.writerow(get_header())
        for i in range(len(test_data_raw)):
            local_result = []
            local_result.append(test_data_raw[i].name)
            local_result.append(test_data_raw[i].description)
            local_result.append(test_data_raw[i].tweet_text)
            local_result.append(test_data_raw[i].link_color)
            local_result.append(test_data_raw[i].sidebar_color)
            local_result.append(test_data_raw[i].label)
            local_result.append(predictions[i])
            local_result.append(correct_predictions[i])
            wr.writerow(local_result)

def get_header():
    header = []
    header.append('name')
    header.append('description')
    header.append('tweet_text')
    header.append('link_color')
    header.append('sidebar_color')
    header.append('true label')
    header.append('predicted label')
    header.append('correct_prediction')
    return header

