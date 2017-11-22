import os
from pathlib import Path
import csv
from PIL import Image
import requests

parent_dir = str(Path().resolve().parent.parent)
inp_file_name = 'gender-classifier-DFE-791531.csv'
source_path = os.path.join(parent_dir, 'Datasets', 'Kaggle', inp_file_name)
# Reading a csv file
data = []
access_features = [5,6,10,13,14,16,17,18,19] 
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
                screen_name = row[14]  # 15th row in original csv file contains the data
                image_url = row[16]
                split_image_url = image_url.split('/')[-1]
                image_extension = split_image_url.split('.')[-1]
                image_name = screen_name + '.' + image_extension

                image_path = os.path.join(parent_dir, 'Datasets', 'Kaggle', 'ProfileImages', image_name)

                file_exists = Path(image_path).is_file()

                if not file_exists:  # Try downloading the image
                    download_count = 0
                    while (not file_exists and download_count < 100):
                        # Try downloading the image
                        f = open(image_path, 'wb')
                        f.write(requests.get(image_url).content)
                        f.close()
                        # Check if thee image is corrupted or not
                        try:
                            Image.open(image_path)
                            file_exists = True  # keep retrying till it can be opened
                        except:
                            file_exists = False

                        download_count = download_count + 1

                    if (download_count == 100 and not file_exists):
                        print 'Failed to download ', image_name, ' after 100 tries. '

                # If the file exists or it could be successfully downloaded from the internet
                if file_exists:  # Append data
                    data_row = []  # OrderedDict()
                    i = 0
                    for i in range(num_features):
                        value = row[access_features[i]]
                        data_row.append(value)

                    data.append(data_row)

        count = count + 1

csvfile.close()
print len(data)


out_file_name = 'Kaggle-filtered.csv'
targ_path = os.path.join(parent_dir, 'Datasets', 'Kaggle', out_file_name)
with open(targ_path, 'w') as outcsv:
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(keys)
    for item in data:
        #Write item to outcsv
        writer.writerow(item)

outcsv.close()
