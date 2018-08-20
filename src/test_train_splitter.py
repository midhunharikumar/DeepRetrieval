import numpy as np
import os
import sys
import glob
from shutil import copyfile
import argparse

test_train_split = 0.7

folder_name = '/Users/midhun/Downloads/dog_cat_dataset/PetIMages/'


def test_train_split(folder_name):

    class_folders = glob.glob(os.path.join(folder_name, '*'))

    class_names = [i.split('/')[-1] for i in class_folders]

    print(class_folders)

    train_folder_path = os.path.join(folder_name, 'train_dir')
    validation_folder_path = os.path.join(folder_name, 'val_dir')

    if not os.path.exists(train_folder_path):
        os.makedirs(train_folder_path)
    if not os.path.exists(validation_folder_path):
        os.makedirs(validation_folder_path)

    # Create the folder structure
    class_folders_train = []
    class_folders_val = []
    for class_name in class_names:
        # Create calss folder in the training directory
        class_folders_train.append(os.path.join(train_folder_path, class_name))
        if not os.path.exists(class_folders_train[-1]):
            os.makedirs(class_folders_train[-1])
        # Create class folder in the validation_directory
        class_folders_val.append(os.path.join(
            validation_folder_path, class_name))
        if not os.path.exists(class_folders_val[-1]):
            os.makedirs(class_folders_val[-1])

    class_files = []

    for idx, class_folder in enumerate(class_folders):
        class_files = glob.glob(os.path.join(class_folder, '*.jpg'))
        for file in class_files[:int(len(class_files) * 0.7)]:
            copyfile(file, os.path.join(
                class_folders_train[idx], file.split('/')[-1]))
        for file in class_files[int(len(class_files) * 0.7):]:
            print(file)
            print(os.path.join(class_folders_val[idx], file.split('/')[-1]))
            copyfile(file, os.path.join(
                class_folders_val[idx], file.split('/')[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split images from folder into test and validation')
    parser.add_argument('--dataset_folder', dest='folder_name',
                        help='Folder with dataset')

    args = parser.parse_args()
    folder_name = args.folder_name
    print(folder_name)
    test_train_split(folder_name)
