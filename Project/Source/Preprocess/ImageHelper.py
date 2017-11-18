import cv2 as cv2
import os as os

current_working_folder = os.path.dirname(os.getcwd())
original_image = 'original.jpg'
output_folder_name = os.path.join(current_working_folder, 'output')


def downsample(folderName, file):
    img = cv2.imread(os.path.join(folderName, file))
    resized_image = cv2.resize(img, (100, 100))
    return resized_image

def get_filenames(folderName, fileExtension = ".jpg"):
    files = []
    for file in os.listdir(folderName):
        if file.endswith(fileExtension):
            files.append(os.path.join('', file))
    return files

def downsample_all(folderName, outputFolderName, fileExtension = ".jpg"):
    filenames = get_filenames(folderName, fileExtension)
    for file in filenames:
        downsampled_image = downsample(folderName, file)
        cv2.imwrite(os.path.join(outputFolderName, 'downsampled_' + file), downsampled_image)

downsample_all('.','.')
