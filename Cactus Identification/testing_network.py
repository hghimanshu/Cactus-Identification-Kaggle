import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import pandas as pd


# load json and create model
json_file = open('Weights/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Weights/weights.h5")
print("Loaded model from disk")
	
    
## Loading the face cascade
path = os.getcwd()
test_url = path + '/data/test/'
test_imgPath = os.listdir(test_url)
print(len(test_imgPath))
data = {'id': [], 'has_cactus': []}

for i, imageName in enumerate(test_imgPath):
    print(i)
    image = cv2.imread(test_url +'/' + imageName)
    image = img_to_array(image)
    image = image/255.0    
    img = image.reshape([1,32,32,3])
    custom = loaded_model.predict(img)
    prediction = custom[0][0]

    if imageName not in data['id']:
        data['id'].append(imageName)
        data['has_cactus'].append(prediction)
        

df = pd.DataFrame(data=data)
df.to_csv(path + '/Final.csv', index=False)
#------------------------------
