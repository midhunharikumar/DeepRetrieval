import keras
from keras.models import Model
from keras.applications import vgg16
from DeepFeatures import DeepFeatures
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import skimage
from keras import optimizers
from parameterLoad import train_parameter_load
import argparse
import os


filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"


def customizedDataAugmentation(x):
    x = x / 255.
    return skimage.transform.resize(x, (224, 224))

train_directory = '/Users/midhun/Downloads/dog_cat_dataset/PetImages/train_dir/'
validation_directory = '/Users/midhun/Downloads/dog_cat_dataset/PetImages/val_dir/'


def fine_train_network(dataset_folder):

    train_directory = os.path.join(dataset_folder, 'train_dir')
    validation_directory = os.path.join(dataset_folder, 'val_dir')
    parameters = train_parameter_load()
    print(parameters)
    df = DeepFeatures(feature_type={
                      'model': 'custom', 'input_layer': 'default', 'output_layer': 'flatten'})
    df.load_custom_model(os.getcwd())
    model = df.getModel()
    for idx, i in enumerate(model.layers):
        model.layers[idx].trainable = True
    print("Model from DF")
    print(model.summary())

    train_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    valid_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(train_directory,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode="categorical",
                                                        shuffle=True)
    valid_generator = valid_datagen.flow_from_directory(validation_directory,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode="categorical",
                                                        shuffle=True)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    adam = keras.optimizers.Adam(
        lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=2
                        )

    df.saveModel(model)


def train_network(dataset_folder):

    train_directory = os.path.join(dataset_folder, 'train_dir')
    validation_directory = os.path.join(dataset_folder, 'val_dir')
    parameters = train_parameter_load()
    print(parameters)
    df = DeepFeatures()

    model = df.getModel()
    model.layers.pop()
    for idx, i in enumerate(model.layers):
        model.layers[idx].trainable = False
    print("Model from DF")
    print(model.summary())

    predictions = Dense(2, activation='softmax')(model.layers[-1].output)

    new_model = Model(input=model.input, output=predictions)

    print(new_model.summary())

    train_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    valid_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(train_directory,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode="categorical",
                                                        shuffle=True)
    valid_generator = valid_datagen.flow_from_directory(validation_directory,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode="categorical",
                                                        shuffle=True)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    new_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    new_model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=2
                            )
    df.saveModel(new_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split images from folder into test and validation')
    parser.add_argument('--dataset_folder', dest='folder_name',
                        help='Folder with dataset')

    args = parser.parse_args()
    folder_name = args.folder_name
    print(folder_name)
    # train_network(folder_name)
    fine_train_network(folder_name)
