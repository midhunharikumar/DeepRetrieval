import keras
from keras.models import Model
from keras.applications import vgg16
from DeepFeatures import DeepFeatures
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
import numpy as np
import skimage
from parameterLoad import train_parameter_load

def customizedDataAugmentation(x):
	x=x/255.
	return skimage.transform.resize(x, (224,224))

train_directory = '/Users/midhun/Downloads/dog_cat_dataset/PetImages/train_dir/'
validation_directory = '/Users/midhun/Downloads/dog_cat_dataset/PetImages/val_dir/'



parameters = train_parameter_load()
print(parameters)

df = DeepFeatures()

model = df.getModel()

print(model.summary())
model.layers.pop()


x = model.output

predictions=Dense(2,activation = 'relu')(x)

model = Model(input=model.input,output = predictions)

print(model.summary())


train_datagen = ImageDataGenerator(
			rescale = 1./255.,
			shear_range =  0.2,
			zoom_range = 0.2,
			horizontal_flip =True)

valid_datagen = ImageDataGenerator(
			rescale = 1./255.,
			shear_range = 0.2,
			zoom_range = 0.2,
			horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_directory,
													target_size=(224, 224))
valid_generator = valid_datagen.flow_from_directory(validation_directory,
													target_size=(224, 224))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.compile(optimizer='rmsprop',    
                loss='categorical_crossentropy', 
                metrics=['accuracy'])


model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)
df.saveModel(model)