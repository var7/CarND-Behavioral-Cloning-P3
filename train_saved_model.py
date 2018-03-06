import os
import csv
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

def get_samples(DATA_FOLDER):
    samples = []
    with open(DATA_FOLDER+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def cull_data(samples, threshold = 0.2):
    ind_to_be_deleted = []
    total_length = len(samples)
    count = 0

    for ind, line in enumerate(samples):
        if line[3] == 'steering':
            continue
        angle = float(line[3])
        if(abs(angle) <= threshold):
            count = count + 1
            if random.random() > 0.3:
                ind_to_be_deleted.append(ind)

    print('original number of samples:', total_length)
    print('total count of samples less than threshold {}:{}'.format \
        (threshold, count))
    print('number of samples less than threshold that are saved: ', \
        0.2 * total_length)
    print('Number of samples to be deleted:', len(ind_to_be_deleted))

    for index in sorted(ind_to_be_deleted, reverse=True):
        del samples[index]

    print('Sample size after deleting:', len(samples))
    return samples

def plot_data(samples, data_name, AWS=False):
    angles = []
    for line in samples:
        if line[3] == 'steering':
            continue
        angle = float(line[3])
        angles.append(angle)
    fig = plt.figure()
    plt.hist(angles, bins='auto')
    plt.title("Angles")
    plt.xlabel("Angle")
    plt.ylabel("Frequency")
    if not(AWS):
        plt.show()
    else:
        fig.savefig(data_name+'.png', bbox_inches='tight')

def generator(samples, img_path, batch_size=32):
    correction = 0.25
    num_samples = len(samples)
    limit_reached = True
    while 1: # Loop forever so the generator never terminates
        count  = 0
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
        for line in batch_samples:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = img_path + filename
            left_filename = line[1].split('/')[-1]
            left_path = img_path + left_filename
            right_filename = line[2].split('/')[-1]
            right_path = img_path + right_filename
            image = cv2.imread(current_path)
            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)
            if image is None:
                print('Incorrect path', current_path)
            else:
                measurement = float(line[3])
                # if limit_reached or abs(measurement) > 0.85:
                #     if abs(measurement) < 0.85:
                #         count = count+1
                #     if count > 10:
                #         limit_reached = False
                images.extend([image, left_image, right_image])
                steering_left = measurement + correction
                steering_right = measurement - correction
                measurements.extend([measurement, steering_left, steering_right])
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
#            print('\ncount', count)
            yield sklearn.utils.shuffle(X_train, y_train)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.models import load_model

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
#    model.add(Dropout(0.50))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(1))
    return model

def train_model(model, model_name, train_generator, validation_generator,\
 train_sample_len, valid_sample_len, epochs=3):

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, \
                samples_per_epoch= (train_sample_len//batch_size)*batch_size, \
                validation_data=validation_generator, \
                nb_val_samples=(valid_sample_len//batch_size)*batch_size, \
                nb_epoch=epochs)
    model.save(model_name+'.h5')
    print('saved model', model_name)
    return history_object

def load_trained_model(weights_path):
   # model = create_model()
   # model.load_weights(weights_path)
   model = load_model(weights_path)
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

DATA_FOLDER = './shiny-data/'
NEW_MODEL_NAME = 'nvidia_datamodelv1-udacity-all-new'
SAVED_MODEL_PATH = './nvidia_datamodel.h5'
IMGPATH = DATA_FOLDER + 'IMG/'

all_samples = get_samples(DATA_FOLDER)
plot_data(all_samples, data_name='shiny_data', AWS=True)
culled_samples = cull_data(all_samples, threshold = 0.5)
plot_data(culled_samples, data_name='shiny_data_culled', AWS=True)

train_samples, validation_samples = train_test_split(culled_samples,test_size=0.2)
train_sample_len = 6*len(train_samples)
valid_sample_len = 6*len(validation_samples)
print('Total number of training samples:', train_sample_len)
print('Total number of validation samples:', valid_sample_len)
batch_size = 32
# compile and train the model using the generator function
train_generator = generator(train_samples, IMGPATH, batch_size=batch_size)
validation_generator = generator(validation_samples, IMGPATH, batch_size=batch_size)

# model = load_trained_model(SAVED_MODEL_PATH)
# # model = create_model()
# train_model(model, NEW_MODEL_NAME, train_generator, validation_generator, \
#     train_sample_len, valid_sample_len, epochs = 3)
