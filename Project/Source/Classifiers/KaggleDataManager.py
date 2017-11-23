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

current_working_folder = os.path.dirname(os.getcwd())
kaggle_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
kaggle_images_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/Images')
train_metadata_filename = 'KaggleTwitter.csv'
serialized_train_metadata_filename = 'KaggleTwitter-600.json'
test_metadata_filename = 'KaggleTwitter.csv'


class KaggleDataManager(object):
    def __init__(self):
        self.__init__(kaggle_files_path + train_metadata_filename)

    def __init__(self, trainingDataFileName):
        self._i = 0
        self.data = []
        #self.read_data_from_csv(trainingDataFileName)
        #clean_data = self.get_clean_kaggle_dataset()
        #self.serialize_data(clean_data)
        self.train = self.deserialize_data()
        print(str(len(self.train)) + " total input examples.")
        self.test = self.train

    def read_data_from_csv(self, trainingDataFileName):
        with open(trainingDataFileName, 'rt') as csvfile:
            # has_header = csv.Sniffer().has_header(csvfile.read(25))
            # csvfile.seek(0)  # rewind
            reader = csv.reader(csvfile, delimiter=',')
            # if has_header:
            next(reader)  # skip header row
            # reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.data.append(KaggleSample(row))
                #if (len(self.data) == 600):
                    #break
        print(len(self.data))

    def serialize_data(self,data):
        filename_to_serialize_into = os.path.join(kaggle_files_path,serialized_train_metadata_filename)
        with open(filename_to_serialize_into, 'wt') as file:
            encoded_data = jsonpickle.encode(data)
            file.write(encoded_data)

    def deserialize_data(self):
        filename_to_serialize_into = os.path.join(kaggle_files_path,serialized_train_metadata_filename)
        print("Reading file:" + filename_to_serialize_into)
        with open(filename_to_serialize_into, 'rt') as file:
            file_data = file.read()
            decoded_data = jsonpickle.decode(file_data)
        return decoded_data

    def get_clean_kaggle_dataset(self):
        goodData = []
        for sample in self.data:
            try:
                img_data = sample.get_image_data()
                goodData.append(sample)
            except:
                print("Omitting:", sample.name)
        return goodData

    def next_batch(self, batch_size):
        batch_data = self.train[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.train)
        return batch_data

class KaggleSample(object):

    def __init__(self):
        pass

    def __init__(self, rowData):
        if rowData is None:
            return
        self.label = 0 if rowData[0] == 'male' else 1
        self.description = rowData[2]
        self.link_color = rowData[3]
        self.name = rowData[4]
        self.profile_image = rowData[5]
        self.sidebar_color = rowData[7]
        self.tweet_text = rowData[8]

    def set(self,dict):
        self.label = dict["label"]
        self.description = dict["description"]
        self.link_color = dict["link_color"]
        self.name = dict["name"]
        self.profile_image = dict["profile_image"]
        self.sidebar_color = dict["sidebar_color"]
        self.tweet_text = dict["tweet_text"]


    def get_image_data(self):
        file_name = self.profile_image.split('/')[-1]
        file_extension = file_name.split('.')[-1]
        full_file_name = os.path.join(kaggle_images_path, 'downsampled_' + self.name + '.' + file_extension)
        image_data = plt.imread(full_file_name, format=file_extension)
        # image_data = image_data.astype(float) / 255
        return image_data


#process_kaggle_dataset()
kdm = KaggleDataManager(kaggle_files_path + train_metadata_filename)
len(kdm.test)
img = kdm.test[0].get_image_data()
print (img)


