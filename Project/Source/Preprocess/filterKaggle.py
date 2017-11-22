import os
from pathlib import Path
import csv
from collections import OrderedDict
import json

parent_dir = str(Path().resolve().parent.parent)
inp_file_name = 'gender-classifier-DFE-791531.csv'
source_path = os.path.join(parent_dir, 'Datasets', 'Kaggle', inp_file_name)
# Reading a csv file
data = []
access_features = range(12) + [13] + range(15, 20) + range(21, 24)
num_features = len(access_features)
keys = [[] for x in range(num_features)]

with open(source_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        # Extracting required features and concatenating them
        if count == 0:

            i = 0
            for i in range(num_features):
                keys[i] = row[access_features[i]]

        else:

            if(row[5] != 'brand'):  # Ignorinng brands
                data_row = []  # OrderedDict()
                i = 0
                for i in range(num_features):
                    value = row[access_features[i]]
                    data_row.append(value)

                data.append(data_row)

        count = count + 1

csvfile.close()


out_file_name = 'KagglewoBrand.csv'
targ_path = os.path.join(parent_dir, 'Datasets', 'Kaggle', out_file_name)
with open(targ_path, 'w') as outcsv:
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(keys)
    for item in data:
        #Write item to outcsv
        writer.writerow(item)

outcsv.close()