from graphSearch import GraphSearch
from DeepFeatures import DeepFeatures
import os
import glob
import json

file_list_name = 'image_index_file.json'
file_list_path = os.path.join(os.getcwd(), file_list_name)


class ImageRetrieval():

    def __init__(self, image_folder_name):
        self.gs = GraphSearch()
        self.folder_name = image_folder_name
        self.files = glob.glob(os.path.join(self.folder_name, '**/*.jpg'))
        self.num_files = len(self.files)
        print("Number of files", len(self.files))
        self.index_file_generator(self.files, force_generation=True)
        self.feature_gen = DeepFeatures(feature_type={
            'model': 'custom', 'input_layer': 'default', 'output_layer': 'fc2'})

    def create_index(self, create_new=False):
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
            self.gs.load_index()

    def get_match(self, image_file):
        image = self.feature_gen.read_image(image_file)
        feature = self.feature_gen.get_feature(image)
        return self.gs.knn(feature.ravel())[0][0]

    def index_file_loader(self):
        with open(file_list_path) as file:
            self.file_list = json.loads(file.read())

    def index_file_generator(self, filenames, force_generation=False):
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
        file_list = {}
        file_list['index'] = []
        for idx, i in enumerate(filenames):
            ''' Strip filepath to only classname and file id.
            This will keep things sane when working across multiple systems.
            As long as the file structure is maintained we should be good '''
            file_list['index'].append(
                "{'" + str(idx) + "':'" + '/'.join(i.split('/')[-2:]) + "'}")
        with open(file_list_path, 'w') as outfile:
            # save to Json file
            json.dump(file_list, outfile)
            print('File List saved to ' + file_list_path)

    def batch_feature(self):
        self.index_file_loader()
        gs = GraphSearch()
        # TODO Remove all hard links.
        print(len(files))
        feature_store = []
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
                image_batch = np.concatenate(
                    (image_batch, image_decoded), axis=0)
            if (not (idx) % (batch_size)) or (idx >= len(file_indexes) - 1):
                print(image_batch.shape)
                feature_store.extend(self.feature_gen.get_feature(image_batch))
                image_batch = []
        print('feature_store shape', np.array(feature_store).shape)
        gs.create_index(np.array(feature_store, np.float32), file_indexes)
        gs.save_index()
        query = gs.knn(feature_store[0])
        print(query)
