import tensorflow as tf
import scipy
from skimage import transform
import matplotlib.pyplot as plt
from keras import optimizers
import keras
from keras.models import Model, model_from_json
from keras.applications import vgg16
import numpy as np
import glob
import os
import argparse
from graphSearch import GraphSearch

# TODO General Docstrings.


class DeepFeatures():

    def __init__(self, feature_type={'model': 'vgg16', 'input_layer': 'default', 'output_layer': 'flatten'}):
        if feature_type['model'] == 'vgg16':
            self.feature_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                                                                input_tensor=None, input_shape=None, pooling=None, classes=1000)
        if feature_type['model'] == 'custom':
            self.load_custom_model(os.getcwd())
        self.graph = tf.get_default_graph()
        self.load_intermediate_model(feature_type['output_layer'])

    def load_custom_model(self, model_folder_name):
        print(model_folder_name)
        json_file = open(os.path.join(model_folder_name, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        self.feature_model = loaded_model

        self.load_intermediate_model('flatten')

    def load_intermediate_model(self, layer_name):
        self.intermediate_layer_model = Model(
            inputs=self.feature_model.input, outputs=self.feature_model.get_layer(layer_name).output)

    def get_feature(self, image):
        with self.graph.as_default():
            feature = self.intermediate_layer_model.predict(
                image, batch_size=image.shape[0])
            print(feature.shape)
        return feature

    def getModel(self):
        return self.feature_model

    def saveModel(self, model):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")

    def read_image(self, file_name):
        image_decoded = scipy.ndimage.imread(
            file_name, flatten=False, mode=None)
        image_decoded = transform.resize(image_decoded, [224, 224, 3])
        image_decoded = np.expand_dims(image_decoded, axis=0)
        return image_decoded

    def train(self):
        # TODO Move training handle here.
        # Training function
        # New module for training
        pass


def batch_feature(folder_name):
    gs = GraphSearch()
    # TODO Remove all hard links.
    print(len(files))
    feature_store = []
    df = DeepFeatures(feature_type={
                      'model': 'custom', 'input_layer': 'default', 'output_layer': 'fc2'})
    # TODO Remove hardcoded value.
    file_indexes = np.random.choice(len(files), 25000)
    print(file_indexes.size)
    # TODO Move batchsize to properties file.
    batch_size = 128
    image_batch = []
    for idx, filenumber in enumerate(file_indexes):
        print(idx)
        image_decoded = scipy.ndimage.imread(
            files[filenumber], flatten=False, mode=None)
        # TODO Move hardcoded values to properties file.
        image_decoded = transform.resize(image_decoded, [224, 224, 3])
        image_decoded = np.expand_dims(image_decoded, axis=0)
        if image_batch == []:
            image_batch = image_decoded
        else:
            image_batch = np.concatenate((image_batch, image_decoded), axis=0)
        if (not (idx) % (batch_size)) or (idx >= len(file_indexes) - 1):
            print(image_batch.shape)
            feature_store.extend(df.get_feature(image_batch))
            image_batch = []
    print('feature_store shape', np.array(feature_store).shape)
    gs.create_index(np.array(feature_store, np.float32), file_indexes)
    gs.save_index()
    query = gs.knn(feature_store[0])
    print(query)


def single_feature():

    gs = GraphSearch()
    # TODO Move to command line arguments
    folder_name = '/Users/midhun/Downloads/kagglecatsanddogs_3367a/PetImages'

    files = glob.glob(os.path.join(folder_name, '**/*.jpg'))
    print(len(files))
    feature_store = []
    df = DeepFeatures()
    file_indexes = np.random.choice(len(files), 10000)
    print(file_indexes.size)
    for idx, filenumber in enumerate(file_indexes):
        print(idx)
        image_decoded = scipy.ndimage.imread(
            files[filenumber], flatten=False, mode=None)
        image_decoded = transform.resize(image_decoded, [224, 224, 3])
        image_decoded = np.expand_dims(image_decoded, axis=0)
        feature_store.append(df.get_feature(image_decoded).ravel())
    print(np.array(feature_store).shape)
    gs.create_index(np.array(feature_store, np.float32), file_indexes)
    gs.save_index()
    query = gs.knn(feature_store[0])
    print(query)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Split images from folder into test and validation')
    parser.add_argument('--dataset_folder', dest='folder_name',
                        help='Folder with dataset', required=True)

    args = parser.parse_args()
    folder_name = args.folder_name
    batch_feature(folder_name)
