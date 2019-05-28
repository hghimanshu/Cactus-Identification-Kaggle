import pandas as pd
import numpy as np
import keras
import cv2
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os


EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (32, 32, 3)
flag = None

path = os.getcwd()

train_url = path + '/data/train/'

train_imgPath = os.listdir(train_url)

path = path + '/train.csv'
data = pd.read_csv(url)
print('\t *********** Summary of Data *************')
print(data.head())

#####################
## Processing Dataset
has_cactus = []
no_cactus = []

has_cactus = data.loc[data['has_cactus'] == 1]['id'].tolist()
no_cactus = data.loc[data['has_cactus'] == 0]['id'].tolist()

print("Total images with cactus :",len(has_cactus))
print("Total images without cactus :",len(no_cactus))

## Generating our Training Set 

data = []
labels = []

for imageName in train_imgPath:
    image = cv2.imread(train_url +'/' + imageName)
    image = img_to_array(image)
    data.append(image)

    if imageName in has_cactus:
        flag = 1
    elif imageName in no_cactus:
        flag = 0
    else:
        flag = None
    
    if flag in range(2):
        labels.append(flag)

data = np.array(data, dtype="float")/255.0
label = np.array(labels, dtype="float")


########################
## Writing the Neural Network
model = Sequential()

## First Layer
model.add(BatchNormalization(input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

## Second Layer

model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


## Third Layer

model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

## Flattening layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy'
    , optimizer=opt
    , metrics=['accuracy']
)

print('\t ********* Model Summary ********** ')
print(model.summary())

#------------------------------
## Start the training    
print('\t ********* Starting the training ********** ')
model.fit(data, label, epochs=100, validation_split=0.3,shuffle=True) #train for all trainset

##########################------------------------############################
# serialize model to JSON
weights_path = path + '/Weights/'

if not os.path.exists(weights_path):
	os.mkdir(weights_path)

model_json = model.to_json()
with open(weights_path + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weights_path + "weights.h5")
print("Saved model to disk")
############################---------------------###############################
