import filterKaggle as k
import urllib
import requests
import os

data = k.data

def is_image(image_content):
    return True

for item in data:
    image_url = item[14]  # GEt the image name
    split_image_url = image_url.split('/')
    out_image_name = split_image_url[-1]
    targ_path = os.path.join(k.parent_dir, 'Datasets', 'Kaggle', 'ProfileImages', out_image_name)

    f = open(targ_path,'wb')
    image_content = requests.get(image_url).content
    if is_image(image_content):
        f.write(image_content)
    f.close()



