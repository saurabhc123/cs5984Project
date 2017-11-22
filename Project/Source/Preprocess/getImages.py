import filterKaggle as k
import urllib
import requests
import os
from pathlib import Path
from PIL import Image

data = k.data

"""
for item in data:
    image_url = item[14]  # GEt the image name
    split_image_url = image_url.split('/')
    out_image_name = split_image_url[-1]
    targ_path = os.path.join(k.parent_dir, 'Datasets', 'Kaggle', 'ProfileImages', out_image_name)

    my_file = Path(targ_path)
    if not my_file.is_file():  # if the file doesn't exist
  
"""
error_urls = []
count = 0
for item in data:
    image_url = item[14]  # GEt the image name
    split_image_url = image_url.split('/')
    out_image_name = split_image_url[-1]
    targ_path = os.path.join(k.parent_dir, 'Datasets', 'Kaggle', 'ProfileImages', out_image_name)
    flag = True

    my_file = Path(targ_path)
    if not my_file.is_file():  # if the file doesn't exist remove the row
        data.remove(item)
        count = count + 1

print(len(data))

print('Number of entries removed : ', count)
"""
# Read the files in the outpur dictionary
f = []
for (dirpath, dirnames, filenames) in os.walk(os.path.join(k.parent_dir, 'Datasets', 'Kaggle', 'ProfileImages')):
    f.extend(filenames)

out_file_name = 'Kaggle-filtered.csv'
targ_path = os.path.join(k.parent_dir, 'Datasets', 'Kaggle', out_file_name)
"""
