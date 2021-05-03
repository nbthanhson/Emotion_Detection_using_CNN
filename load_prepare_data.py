from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np

#Load dataset
path = '/content/gdrive/MyDrive/Emotion Detection/Data/fer2013.csv'
df = pd.read_csv(path)

#Split data into training and testing set
train_x, train_y, test_x, test_y = [], [], [], []
for index, row in df.iterrows():
    val = row['pixels'].split(' ')
    try:
        if 'Training' in row['Usage']:
            train_x.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            test_x.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print('Error occured at index: {} and row: {}'.format(index, row))

from tensorflow import keras
from keras.utils import np_utils

train_x = np.array(train_x, 'float32')
train_y = np.array(train_y, 'float32')
test_x = np.array(test_x, 'float32')
test_y = np.array(test_y, 'float32')

train_y = np_utils.to_categorical(train_y, num_classes = 7)
test_y = np_utils.to_categorical(test_y, num_classes = 7)

train_x /= 255.0
train_x -= 0.5
train_x *= 2.0
test_x /= 255.0
test_x -= 0.5
test_x *= 2.0

train_x = train_x.reshape(train_x.shape[0], 48, 48, 1)
test_x = test_x.reshape(test_x.shape[0], 48, 48, 1)

#Split training set into training and validation data
length_samples = len(train_x)
length_train_samples = int((1 - 0.2) * length_samples)

train_data = train_x[:length_train_samples]
train_label = train_y[:length_train_samples]

val_x = train_x[length_train_samples:]
val_y = train_y[length_train_samples:]
val_data = (val_x, val_y)

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
                        featurewise_center = False, 
                        featurewise_std_normalization = False,
                        rotation_range = 10,
                        width_shift_range = 0.1,
                        height_shift_range = 0.1,
                        zoom_range = 0.1,
                        horizontal_flip = True)
