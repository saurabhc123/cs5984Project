import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
import scipy as sc
import ConvHelper
import jsonpickle
import csv as csv
import datetime
import os as os
import random
import Placeholders
import copyreg
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import webcolors as wc

current_working_folder = os.path.dirname(os.getcwd())
kaggle_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
kaggle_images_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/Images')
train_metadata_filename = 'Kaggle-str-process.csv'
serialized_train_metadata_filename = 'KaggleTwitter.json'
test_metadata_filename = 'KaggleTwitter.csv'


x = Placeholders.x

class KaggleDataManager(object):
    def __init__(self):
        self.__init__(kaggle_files_path + train_metadata_filename)

    def __init__(self, trainingDataFileName, sess, fc7, word_vec):
        self._i = 0
        self.sess = sess
        self.fc7 = fc7
        self.word_vec = word_vec
        self.data = []
        self.read_data_from_csv(trainingDataFileName)

        clean_kaggle_dataset = self.get_clean_kaggle_dataset()

        clean_data = train_test_split(clean_kaggle_dataset, test_size = 0.3, random_state = 45)
        self.train_raw = clean_data[0]
        self.test_raw, self.validation_raw = train_test_split(clean_data[1], test_size=0.5,
                                                              random_state = 13)

        normalized_kaggle_instance = KaggleDataSet(clean_kaggle_dataset)
        normalized_kaggle_dataset = normalized_kaggle_instance.get_normalized_dataset()
        train, test = train_test_split(normalized_kaggle_dataset, test_size = 0.3, random_state = 45)
        self.train = KaggleDataSet(train[:,:-1], train[:,-1])
        print(str(len(self.train.features)) + " total input training examples.")
        test , validation = train_test_split(test, test_size = 0.5, random_state = 13)
        self.test = KaggleDataSet(test[:,:-1], test[:,-1])
        self.validation = KaggleDataSet(validation[:,:-1], validation[:,-1])

    def read_data_from_csv(self, trainingDataFileName):
        with open(trainingDataFileName, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip header row
            for row in reader:
                self.data.append(KaggleSample(row, self.sess, self.fc7, self.word_vec))
        print(len(self.data))

    def get_clean_kaggle_dataset(self):
        goodData = []
        for sample in self.data:
            try:
                img_data = sample.get_image_data()
                goodData.append(sample)
            except:
                pass
                #print("Omitting:", sample.name)
        return goodData



    def next_batch(self, batch_size):
        batch_features = self.train.features[self._i:self._i+batch_size][0:Placeholders.feature_width - 50]
        batch_labels = self.train.features[self._i:self._i + batch_size][Placeholders.feature_width:-1]
        self._i = (self._i + batch_size) % len(self.train.features)
        return batch_features, batch_labels

class KaggleDataSet(object):

    def __init__(self, data, labels = None):
        if(labels is None):
            results = self.get_features_and_labels(data)
            self.features = results[0]
            self.labels = results[1]
        else:
            self.from_features_and_labels(data, labels)

    def from_features_and_labels(self, features, labels):
        self.features = features
        self.labels = labels


    def get_features_and_labels(self, batch_data):
        labels = np.array(list(map(lambda x: x.label, batch_data)))
        labels = labels.reshape(-1, 1)
        features = np.array(list(map(lambda x: x.get_feature_from_sample(), batch_data))).reshape(-1, Placeholders.feature_width)

        # print(labels.shape , images.shape)
        labels = self.one_hot(np.hstack([int(d) for d in labels]), Placeholders.n_classes)
        # print(features.shape , labels.shape)
        return np.array(features), np.array(labels)

    def get_normalized_dataset(self):
        index = 0
        normalized_features = np.array([]).reshape((len(self.features), 0))
        normalized_features =  np.hstack((normalized_features, preprocessing.scale(self.features[:,index : Placeholders.img_feature_width])))
        index = normalized_features.shape[1]
        normalized_features = np.hstack((normalized_features, preprocessing.scale(self.features[:,index : index + Placeholders.word_vec_length])))
        index = normalized_features.shape[1]
        features_to_scale = self.features[:,index : index + Placeholders.profile_color_feature_length]
        normalized_features = np.hstack((normalized_features, preprocessing.scale(features_to_scale)))
        index = normalized_features.shape[1]
        normalized_features = np.hstack((normalized_features, self.labels))
        return normalized_features

    def one_hot(self, vec, vals = Placeholders.n_classes):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

class KaggleSample(object):
    def __init__(self, rowData, sess, fc7, word_vec):
        self.label = int(0) if rowData[0] == 'male' else int(1)
        self.description = rowData[2]
        self.link_color = rowData[3]
        self.name = rowData[4]
        self.profile_image = rowData[5]
        self.sidebar_color = rowData[7]
        self.tweet_text = rowData[8]
        self.sess = sess
        self.fc7 = fc7
        self.word_vec = word_vec

    def get_image_data(self):
        file_name = self.profile_image.split('/')[-1]
        file_extension = file_name.split('.')[-1]
        full_file_name = os.path.join(kaggle_images_path,'downsampled_' + self.name + '.'+ file_extension)
        image_data = plt.imread(full_file_name, format=file_extension)
        #image_data = image_data.astype(float) / 255
        return image_data

    def get_fc7_representation(self, sample):
        image = np.array(sample).reshape((-1, Placeholders.img_dim, Placeholders.img_dim, Placeholders.n_channels))
        fc7rep = self.sess.run(self.fc7, feed_dict={x: image, Placeholders.adience_keep_prob: 1.0})
        return np.array(fc7rep)



    def get_feature_from_sample(self):
        features = np.array([]).reshape((1, 0))
        fc7_x = self.get_fc7_representation(self.get_image_data())
        desc_word_vector = self.word_vec.get_sentence_vector(self.description)
        tweet_word_vector = self.word_vec.get_sentence_vector(self.tweet_text)
        sidebar_feature = self.hex_to_rgb(self.sidebar_color)
        link_color_feature = self.hex_to_rgb(self.link_color)
        features = self.compose_features(desc_word_vector, fc7_x, features, link_color_feature, sidebar_feature,
                                         tweet_word_vector)
        return features

    def compose_features(self, desc_word_vector, fc7_x, features, link_color_feature, sidebar_feature, tweet_word_vector):
        features = np.hstack((features, fc7_x))
        features = np.hstack((features, desc_word_vector))
        features = np.hstack((features, tweet_word_vector))
        features = np.hstack((features, sidebar_feature))
        features = np.hstack((features, link_color_feature))
        features = features.reshape(-1, Placeholders.feature_width)
        return features



    def hex_to_rgb(self, hex_color):
        base_color = np.zeros(shape=[1, 3])
        if (hex_color is None):
            return base_color
        try:
            color = np.array(wc.hex_to_rgb("#" + hex_color))
            return color.reshape(1, 3)
        except:
            pass
        return base_color




#process_kaggle_dataset()
#kdm = KaggleDataManager(kaggle_files_path + train_metadata_filename)
#len(kdm.test)
# img = kdm.test[0].get_image_data()


