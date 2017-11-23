import cv2 as cv2
import os as os

current_working_folder = os.path.dirname(os.getcwd())
original_image = 'original.jpg'
output_folder_name = os.path.join(current_working_folder, 'downsampled')


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

def get_subfolders(folderName):
    dirlist = [ os.path.join(folderName, item) for item in os.listdir(folderName) if os.path.isdir(os.path.join(folderName, item)) ]
    return dirlist

def downsample_all(folderName, outputFolderName, fileExtension = ".jpg"):
    filenames = get_filenames(folderName, fileExtension)
    for file in filenames:
        try:
            downsampled_image = downsample(folderName, file)
            cv2.imwrite(os.path.join(folderName, 'downsampled_' + file), downsampled_image)
        except:
            print("Error with file:",file)

#downsample_all('.',output_folder_name)

#The following code could be used to downsample images in a list of directories. For a single directory, just call the
# downsample_all(folder,output_folder_name) method with the foldername parameter.
# folders = get_subfolders('/Users/saur6410/Google Drive/VT/Dissertation/cs5984Project/Project/Datasets/Kaggle/Images')
# for folder in folders:
#     print "processing:", folder
#     downsample_all(folder,output_folder_name)
#     break;

folder = '/Users/saur6410/Google Drive/VT/Dissertation/cs5984Project/Project/Datasets/Kaggle/Images'
downsample_all(folder,output_folder_name, fileExtension = ".png")
