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
import ImagePlaceholders
import Placeholders
import ImageDataManager as DataManager
import word2Vec as w2v
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


model_folder_name = "models/adience"
model_filename = os.path.join(model_folder_name,"image_model.ckpt")
STEPS = 50
MINIBATCH_SIZE = 100


#Kaggle data
current_working_folder = os.path.dirname(os.getcwd())
kaggle_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
kaggle_images_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/Images')
train_metadata_filename = 'Kaggle-str-process.csv'
test_metadata_filename = 'Kaggle-str-process.csv'
today = datetime.datetime.now()
format = "%d_%m_%Y_%H_%M_%S"
run_folder = today.strftime(format)


x_main = tf.placeholder(tf.float32, shape=[None, ImagePlaceholders.feature_width])
keep_prob = Placeholders.keep_prob

train_loss_results = []
test_loss_results = []
validation_loss_results = []

train_accuracy_results = []
test_accuracy_results = []
validation_accuracy_results = []

train_precision_results = []
test_precision_results = []
validation_precision_results = []

train_recall_results = []
test_recall_results = []
validation_recall_results = []

train_f1_results = []
test_f1_results = []
validation_f1_results = []






x = ImagePlaceholders.x
y_ = Placeholders.y_

def test(sess, accuracy, test,fc7, word_vec, y_conv, correct_prediction,loss, kdm, epoch, datasetType = "Test"):
    print ("Starting Test")
    number_of_test_batches = 5
    number_of_samples_per_batch = 10
    total_samples = number_of_test_batches * number_of_samples_per_batch
    random_index = random.randint(0,len(test.features) - total_samples)

    #print len(adience.test.images)
    data = test
    batch_features = data.features[random_index:random_index+total_samples]
    batch_labels = data.labels[random_index:random_index+total_samples]
    batch_x = batch_features.reshape(-1, ImagePlaceholders.feature_width)

    acc = sess.run(accuracy, feed_dict={x_main: batch_x, y_: batch_labels,
                                     keep_prob: 1.0})
    print (datasetType + " Accuracy: {:.4}%".format(acc * 100))
    mse = loss.eval(feed_dict={x_main: batch_x, y_: batch_labels, keep_prob: 1.0})
    predictions = np.array(sess.run(tf.argmax(y_conv, 1), feed_dict={x_main: batch_x, y_: batch_labels, keep_prob: 1.0}))
    correct_predictions = np.array(sess.run(correct_prediction, feed_dict={x_main: batch_x, y_: batch_labels, keep_prob: 1.0}))
    raw_data = kdm.test_raw if datasetType == "Test" else (kdm.train_raw if datasetType == "Train" else kdm.validation_raw)
    write_results_to_file(mse, acc * 100, raw_data, predictions, correct_predictions, epoch, datasetType)
    return acc * 100


def test_all(sess, accuracy, test,fc7, word_vec, y_conv, correct_prediction,loss, kdm, epoch = 0, datasetType = "Test"):
    print ("Starting test on all data:" + datasetType)

    #print len(adience.test.images)
    data = test
    #batch_features =  data.features
    #batch_labels = data.labels
    batch_features = data.features[:, :ImagePlaceholders.feature_width]
    batch_labels = one_hot(data.features[:, ImagePlaceholders.feature_width])
    #batch_x = batch_features#batch_features.reshape(-1, Placeholders.feature_width)

    features = batch_features[:, :ImagePlaceholders.feature_width]
    # print(batch[0][:,Placeholders.feature_width])
    labels = batch_labels
    # print("Features shape:")
    # print(features.shape)
    # print(labels.shape)
    batch_x = features
    rnn_features = np.array(features[:, ImagePlaceholders.img_feature_width + ImagePlaceholders.profile_color_feature_length:]) \
        .reshape((-1, ImagePlaceholders.n_steps, ImagePlaceholders.n_inputs))

    other_features = features[:, :ImagePlaceholders.img_feature_width + ImagePlaceholders.profile_color_feature_length]
    acc = sess.run(accuracy, feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                                        ImagePlaceholders.rnn_other_features: other_features,
                                        y_: labels,
                                        keep_prob: 1.0})
    print (datasetType + " Accuracy: {:.4}%".format(acc * 100))
    mse = loss.eval(feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                               ImagePlaceholders.rnn_other_features: other_features,
                               y_: labels,
                               keep_prob: 1.0})
    print(datasetType + " Loss: {:.4}".format(mse))
    predictions = np.array(sess.run(tf.argmax(y_conv, 1),
                            feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                                       ImagePlaceholders.rnn_other_features: other_features,
                                       y_: labels,
                                       keep_prob: 1.0}))
    correct_predictions = np.array(sess.run(correct_prediction,
                                    feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                                               ImagePlaceholders.rnn_other_features: other_features,
                                               y_: labels,
                                               keep_prob: 1.0}))
    f1_predictions = np.array(predictions)
    f1_truelabels = np.argmax(batch_labels, 1)
    print(f1_truelabels)
    print(f1_predictions)
    f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
    precision = precision_score(f1_truelabels, f1_predictions, average='macro')
    recall = recall_score(f1_truelabels, f1_predictions, average='macro')
    print(datasetType + " Precision: {:.4}%".format(precision)+ " Recall: {:.4}%".format(recall)+ " F1: {:.4}%".format(f1score))
    raw_data = kdm.test_raw if datasetType == "Test" else (kdm.train_raw if datasetType == "Train" else kdm.validation_raw)
    compose_metrics(datasetType, mse, acc * 100, precision, recall, f1score)
    write_results_to_file(mse, acc * 100, raw_data, predictions, correct_predictions, epoch, datasetType,f1score, precision, recall)
    return acc*100

def compose_metrics(datasetType, mse, accuracy, precision, recall, f1score):
    if datasetType == 'Train':
        train_loss_results.append(mse)
        train_accuracy_results.append(accuracy)
        train_precision_results.append(precision)
        train_recall_results.append(recall)
        train_f1_results.append(f1score)
    elif datasetType == 'Validation':
        validation_loss_results.append(mse)
        validation_accuracy_results.append(accuracy)
        validation_precision_results.append(precision)
        validation_recall_results.append(recall)
        validation_f1_results.append(f1score)
    else:
        test_loss_results.append(mse)
        test_accuracy_results.append(accuracy)
        test_precision_results.append(precision)
        test_recall_results.append(recall)
        test_f1_results.append(f1score)

def write_metrics(datasetType, wr):
    if datasetType == 'Train':
        wr.writerow(train_loss_results)
        wr.writerow(train_accuracy_results)
        wr.writerow(train_precision_results)
        wr.writerow(train_recall_results)
        wr.writerow(train_f1_results)
    elif datasetType == 'Validation':
        wr.writerow(validation_loss_results)
        wr.writerow(validation_accuracy_results)
        wr.writerow(validation_precision_results)
        wr.writerow(validation_recall_results)
        wr.writerow(validation_f1_results)
    else:
        wr.writerow(test_loss_results)
        wr.writerow(test_accuracy_results)
        wr.writerow(test_precision_results)
        wr.writerow(test_recall_results)
        wr.writerow(test_f1_results)

# Do the default

def train(sess, retrain, fc7):
    output_folder = os.path.join(current_working_folder,"Project/output")
    output_folder = os.path.join(output_folder, run_folder)
    if not os.path.exists(output_folder):
        print("Creating folder: " + output_folder)
        os.makedirs(output_folder)
    else:
        print("Folder exists: " + output_folder)

    word_vec = w2v.word2vec()
    kdm = DataManager.KaggleRNNDataManager(kaggle_files_path + train_metadata_filename, sess, fc7, word_vec)



    all_features = ImagePlaceholders.rnn_other_features

    image_fc1 = tf.nn.dropout(all_features, keep_prob=keep_prob)
    image_logits = tf.layers.dense(image_fc1, ImagePlaceholders.n_classes, activation=tf.nn.relu)
    #fully_connected1_dropout = tf.nn.dropout(output_layer, keep_prob=keep_prob)
    image_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= image_logits, labels=y_)

    image_loss = tf.reduce_mean(image_cross_entropy)
    image_train_step = tf.train.AdamOptimizer(1e-4).minimize(image_loss)

    image_correct_prediction = tf.equal(tf.argmax(image_logits, 1), tf.argmax(y_, 1))
    image_accuracy = tf.reduce_mean(tf.cast(image_correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    STEPS = 500
    MINIBATCH_SIZE = 50


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
            for batch_count in range(int(len(kdm.train.features)/MINIBATCH_SIZE)):
                batch = kdm.next_batch(MINIBATCH_SIZE)
                features = batch[0][:,:ImagePlaceholders.feature_width]
                labels = one_hot(batch[0][:,ImagePlaceholders.feature_width])
                rnn_features = np.array(features[:, ImagePlaceholders.img_feature_width + ImagePlaceholders.profile_color_feature_length:])\
                                .reshape((-1, ImagePlaceholders.n_steps, ImagePlaceholders.n_inputs))
                other_features = features[:, :ImagePlaceholders.img_feature_width + ImagePlaceholders.profile_color_feature_length]
                sess.run(image_train_step, feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                                                      ImagePlaceholders.rnn_other_features : other_features,
                                                y_: labels,
                                                keep_prob: 0.75})
            if(epoch%10 == 0):
                mse = image_loss.eval(feed_dict={ImagePlaceholders.rnn_X: rnn_features,
                                                 ImagePlaceholders.rnn_other_features : other_features,
                                            y_: labels,
                                            keep_prob: 1.0})
                print("Iter " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(mse))
                train_accuracy = test_all(sess, image_accuracy, kdm.train, fc7, word_vec, image_logits, image_correct_prediction, image_loss, kdm, epoch, datasetType="Train")
                validation_accuracy = test_all(sess, image_accuracy, kdm.validation, fc7, word_vec, image_logits, image_correct_prediction, image_loss, kdm, epoch, datasetType="Validation")
                if (validation_accuracy > ImagePlaceholders.best_accuracy_so_far):
                    ImagePlaceholders.best_accuracy_so_far = validation_accuracy
                    test_all(sess, image_accuracy, kdm.test, fc7, word_vec, image_logits, image_correct_prediction, image_loss, kdm, epoch)
                elif (train_accuracy > 70):
                    test_all(sess, image_accuracy, kdm.test, fc7, word_vec, image_logits, image_correct_prediction, image_loss, kdm, epoch)
                np.random.shuffle(kdm.train.features)
        test_all(sess, image_accuracy, kdm.test, fc7, word_vec, image_logits, image_correct_prediction, image_loss, kdm)
        save_path = saver.save(sess, model_filename)
        print("Model saved in file: %s" % save_path)
    return image_logits

def one_hot(vec, vals = ImagePlaceholders.n_classes):
    n = len(vec)
    vec = [int(val) for val in vec]
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def write_results_to_file(loss, accuracy, test_data_raw, predictions, correct_predictions, epoch, datasetType, f1score = 0.0, precision = 0.0, recall = 0.0):
    today = datetime.datetime.now()
    format = "_%d_%m_%Y_%H_%M_%S"
    filename = "output/" + run_folder + "/" + datasetType + "_Image_"+ today.strftime(format) + "_Iteration_" + str(epoch)  + "_Accuracy_" + str(round(accuracy, 2)) + ".csv"
    with open(filename, 'wt') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Loss = " + str(loss)])
        wr.writerow([datasetType + " Accuracy = " + str(accuracy)])
        wr.writerow(["Precision = " + str(precision)])
        wr.writerow(["Recall = " + str(recall)])
        wr.writerow(["F1 = " + str(f1score)])
        write_metrics(datasetType, wr)
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

