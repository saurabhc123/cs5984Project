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

current_working_folder = os.path.dirname(os.getcwd())
kaggle_files_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/')
kaggle_images_path = os.path.join(current_working_folder, 'Project/Datasets/Kaggle/Images')
train_metadata_filename = 'KaggleTwitter.csv'
test_metadata_filename = 'KaggleTwitter.csv'


class KaggleDataManager(object):
    def __init__(self, trainingDataFileName):
        data = []
        with open(trainingDataFileName, 'rt') as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(25))
            csvfile.seek(0)  # rewind
            reader = csv.reader(csvfile, delimiter=',')
            if has_header:
                next(reader)  # skip header row
            #reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(KaggleSample(row))
                if(len(data) == 5):
                    break
        print(len(data))
        self.train = data
        self.test = data





class KaggleSample(object):
    def __init__(self, rowData):
        self.label = 0.0 if rowData[0] == 'male' else 1.0
        self.profile_description = rowData[2]
        self.link_color = rowData[3]
        self.name = rowData[4]
        self.profile_image = rowData[5]
        self.sidebar_color = rowData[7]
        self.tweet_text = rowData[8]

    def get_image_data(self):
        file_name = self.profile_image.split('/')[-1]
        file_extension = file_name.split('.')[-1]
        full_file_name = os.path.join(kaggle_images_path,'downsampled_' + self.name + '.'+ file_extension)
        image_data = plt.imread(full_file_name, format=file_extension)
        #image_data = image_data.astype(float) / 255
        return image_data



kdm = KaggleDataManager(kaggle_files_path + train_metadata_filename)
len(kdm.test)
img = kdm.test[0].get_image_data()


