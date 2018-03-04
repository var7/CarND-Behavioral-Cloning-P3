import os
import csv
import matplotlib.pyplot as plt
plt.switch_backend('agg')
DATA_FOLDER = './newbestdata/'
NEW_MODEL_NAME = 'nvidia_datamodelv2'
SAVED_MODEL_PATH = './models/nvidia_datamodel.h5'

samples = []
with open(DATA_FOLDER+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(samples))
print(len(train_samples))

import cv2
import numpy as np
import sklearn

IMGPATH = DATA_FOLDER + 'IMG/'

def generator(samples, batch_size=32):
    correction = 0.5
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
        for line in batch_samples:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = IMGPATH + filename
            left_filename = line[1].split('/')[-1]
            left_path = IMGPATH + left_filename
            right_filename = line[2].split('/')[-1]
            right_path = IMGPATH + right_filename
            image = cv2.imread(current_path)
            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)
            if image is None:
                print('Incorrect path', current_path)
            else:
                measurement = float(line[3])
                images.append(image)
                measurements.append(measurement)
                images.extend([image, left_image, right_image])
                measurement = float(line[3])
                steering_left = measurement + correction
                steering_right = measurement - correction
                measurements.extend([measurement, steering_left, steering_right])
                if measurement > 0.9 or measurement < -0.9:
                    left_image_flipped = np.fliplr(left_image)
                    image_flipped = np.fliplr(image)
                    right_image_flipped = np.fliplr(right_image)
                    measurement_flipped = -measurement
                    steering_left_flipped = -steering_left
                    steering_right_flipped = -steering_right
                    images.extend([image_flipped, left_image_flipped, \
                        right_image_flipped])
                    measurements.extend([measurement_flipped, steering_left_flipped, \
                    steering_right_flipped])
        # trim image to only see section with road

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24,5,5,subsample=(2,2),activation = "elu"))
    model.add(Conv2D(36,5,5,subsample=(2,2),activation="elu"))
    #model.add(Dropout(0.25))
    model.add(Conv2D(48,5,5,subsample=(2,2), activation='elu'))
    model.add(Conv2D(64,3,3,activation="elu"))
    model.add(Conv2D(64,3,3,activation="elu"))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.50))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(1))
    return model

def train_model(model, model_name):
    model.compile(loss='mse', optimizer='adam')
    samples =  3*len(train_samples)
    history_object = model.fit_generator(train_generator, \
                samples_per_epoch= (samples//batch_size)*batch_size, \
                validation_data=validation_generator, \
                nb_val_samples=(len(validation_samples)//batch_size)*batch_size, \
                nb_epoch=3)

    model.save(model_name+'.h5')
    print('saved model')
    return history_object

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model

def training_plots(history_object, model_name):
    fig = plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig(model_name+'.png', bbox_inches='tight')

saved_model = load_trained_model(SAVED_MODEL_PATH)
train_model(saved_model, NEW_MODEL_NAME)
