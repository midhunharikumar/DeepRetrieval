
from graphSearch import GraphSearch
from DeepFeatures import DeepFeatures
import os
import glob
import json
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
import argparse
import numpy as np
import scipy
from skimage import transform
<<<<<<< HEAD
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3

file_list_name = 'image_index_file.json'
file_list_path = os.path.join(os.getcwd(), file_list_name)


class ImageRetrieval():

    def __init__(self, image_folder_name):
        """ Returns an ImageRetrieval object.
        Image retrieval takes a folder and catalogues all images within it. The
        feature type is hard coded and can be changed if required. The feature 
        extraction module contains more information of feature settings.

        This module acts as the main function for executing the image retrieval engine.

        Args - image_folder_name : Folder name of images used for index as well
                                   as retrieval.

        """
        self.gs = GraphSearch()
<<<<<<< HEAD
        # TODO convert this to argument and move defaults to the parameters
        # file.
        self.feature_gen = DeepFeatures(feature_type={
            'model': 'custom', 'input_layer': 'default', 'output_layer': 'fc2'})
=======
        self.feature_gen = DeepFeatures(feature_type={
            'model': 'custom', 'input_layer': 'default', 'output_layer': 'fc2'})

>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        self.folder_name = image_folder_name
        self.files = glob.glob(os.path.join(self.folder_name, '**/*.jpg'))
        self.num_files = len(self.files)
        print("Number of files", len(self.files))
<<<<<<< HEAD
=======
<<<<<<< HEAD
        self.index_file_generator(self.files, force_generation=True)
        self.feature_gen = DeepFeatures(feature_type={
            'model': 'custom', 'input_layer': 'default', 'output_layer': 'fc2'})
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3

    def create_index(self, create_new=False):
        """ Loads the image index from disk if it is available.

        Args -  create_new : If True creates a new index using images in input folder.

        """
        self.feature_store = []
        if create_new:
            # Create the index file before creating the features and generating
            # the index.
            self.index_file_generator(self.files, force_generation=True)
            for idx, file_name in enumerate(self.files):
                image = self.feature_gen.read_image(file_name)
                feature = self.feature_gen.get_feature(image)
                self.feature_store.append(feature.ravel())
            self.gs.create_index(np.array(feature_store, np.float32),
                                 np.arange(len(self.num_files)))
            self.gs.save_index()
        else:
            # Load index from disk if available.
            self.gs.load_index()

    def get_match(self, image_file):
        """ Retrieve closest matching image given an input image_file

            Args - image_file : Query image file.

            return - match_idx : Index id of matched image.
                     self.files[match_idx[0]] : Filename of Matched image.
        """
        image = self.feature_gen.read_image(image_file)
        feature = self.feature_gen.get_feature(image)
<<<<<<< HEAD
        match_idx = self.gs.knn(feature.ravel())[0][0]
        print("Match id", match_idx)
        return match_idx, self.files[match_idx[0]]

    def index_file_loader(self):
        """ Loads the image file list index from disk.
            Image list index ensures that file modifications do not affect indexing.
            Index files need to be generated.
        """
        with open(file_list_path) as file:
            self.file_list = json.loads(file.read())
=======
<<<<<<< HEAD
        return self.gs.knn(feature.ravel())[0][0]
=======
        match_idx = self.gs.knn(feature.ravel())[0][0]
        print("Match id", match_idx)
        return match_idx, self.files[match_idx[0]]
>>>>>>>  Adding changes

    def index_file_loader(self):
        with open(file_list_path) as file:
            self.file_list = json.loads(file.read())
<<<<<<< HEAD

    def index_file_generator(self, filenames, force_generation=False):
=======
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        self.files = [os.path.join(self.folder_name, self.file_list['index'][
            str(idx)]) for idx, i in enumerate(self.file_list['index'])]
        self.num_files = len(self.files)

    def index_file_generator(self, filenames=None, force_generation=False):
<<<<<<< HEAD
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        ''' Function creates a json file to keep track of index keys that
        correspond to particular images that are indexed. This is nesseary
        to ensure that indexes created on one host can be reused on another
        where the file structure is modified or has been tampered with. As long
        as folder structure remains the same this indexing will hold. Missing
        files can then be restructured. This index can later be moved to a datbase
        entry.

        Args :
                filenames -> Names of the input files
                force_generation -> force a new index file creation even if an
                                    index file exists.
        '''
        if os.path.isfile(file_list_path) and not force_generation:
            print('Index File found skipping file generation')
            exit()
        print('Generating file list.')
<<<<<<< HEAD
=======
<<<<<<< HEAD
        file_list = {}
        file_list['index'] = []
=======
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        if filenames == None:
            filenames = self.files
        file_list = {}
        file_list['index'] = {}
<<<<<<< HEAD
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        for idx, i in enumerate(filenames):
            ''' Strip filepath to only classname and file id.
            This will keep things sane when working across multiple systems.
            As long as the file structure is maintained we should be good '''
<<<<<<< HEAD
            file_list['index'][str(idx)] = '/'.join(i.split('/')[-2:])
=======
<<<<<<< HEAD
            file_list['index'].append(
                "{'" + str(idx) + "':'" + '/'.join(i.split('/')[-2:]) + "'}")
=======
            file_list['index'][str(idx)] = '/'.join(i.split('/')[-2:])
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        with open(file_list_path, 'w') as outfile:
            # save to Json file
            json.dump(file_list, outfile)
            print('File List saved to ' + file_list_path)

<<<<<<< HEAD
    def batch_feature_store(self):
        """ Function performs batch feature generation for indexing.
        """
=======
<<<<<<< HEAD
    def batch_feature(self):
        self.index_file_loader()
        gs = GraphSearch()
        # TODO Remove all hard links.
        print(len(files))
        feature_store = []
        # TODO Remove hardcoded value.
        file_indexes = np.random.choice(len(files), 25000)
=======
    def batch_feature_store(self):
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        self.index_file_loader()
        gs = GraphSearch()
        # TODO Remove all hard links.
        print(len(self.files))
        feature_store = []
        # TODO Remove hardcoded value.
        file_indexes = np.random.choice(len(self.files), self.num_files)
<<<<<<< HEAD
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
        print(file_indexes.size)
        # TODO Move batchsize to properties file.
        batch_size = 128
        image_batch = []
        for idx, filenumber in enumerate(file_indexes):
            print(idx)
            image_decoded = scipy.ndimage.imread(
<<<<<<< HEAD
                self.files[filenumber], flatten=False, mode=None)
=======
<<<<<<< HEAD
                files[filenumber], flatten=False, mode=None)
=======
                self.files[filenumber], flatten=False, mode=None)
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
            # TODO Move hardcoded values to properties file.
            image_decoded = transform.resize(image_decoded, [224, 224, 3])
            image_decoded = np.expand_dims(image_decoded, axis=0)
            if image_batch == []:
                image_batch = image_decoded
            else:
                image_batch = np.concatenate(
                    (image_batch, image_decoded), axis=0)
            if (not (idx) % (batch_size)) or (idx >= len(file_indexes) - 1):
                print(image_batch.shape)
                feature_store.extend(self.feature_gen.get_feature(image_batch))
                image_batch = []
        print('feature_store shape', np.array(feature_store).shape)
        gs.create_index(np.array(feature_store, np.float32), file_indexes)
        gs.save_index()
<<<<<<< HEAD
=======
        query = gs.knn(feature_store[0])
        print(query)
<<<<<<< HEAD
=======

>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split images from folder into test and validation')
    parser.add_argument('--dataset_folder', dest='folder_name',
                        help='Folder with dataset')
    parser.add_argument(
        '--create_index_file', action='store_true', dest='create_index')
    parser.add_argument(
        '--generate_image_index', action='store_true', dest='generate_features')
    args = parser.parse_args()
    folder_name = args.folder_name
    print(folder_name)
    ig = ImageRetrieval(folder_name)
    if args.create_index:
        ig.index_file_generator(force_generation=True)
    if args.generate_features:
        ig.batch_feature_store()
    ig.index_file_loader()
<<<<<<< HEAD
=======
>>>>>>>  Adding changes
>>>>>>> c5b3a9dac6d0789cc8005df728bc69dff4b455b3
