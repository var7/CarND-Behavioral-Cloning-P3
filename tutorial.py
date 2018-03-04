import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

DATA_FOLDER = './bestdata/'
NEW_MODEL_NAME = 'nvidia_datamodel-tutv2'
SAVED_MODEL_PATH = './models/nvidia_datamodel.h5'

lines = []

IMGPATH = DATA_FOLDER + 'IMG/'
with open(DATA_FOLDER+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

print(len(lines))
correction = 0.2
images = []
measurements = []
limit_reached = True
count = 0
for line in lines:
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
		if limit_reached or abs(measurement) > 1:
			if abs(measurement) < 1:
				count = count+1
			if count > 2000:
				limit_reached = False
			images.extend([image, left_image, right_image])
			steering_left = measurement + correction
			steering_right = measurement - correction
			measurements.extend([measurement, steering_left, steering_right])
			image_flipped = np.fliplr(image)
			left_image_flipped = np.fliplr(left_image)
			right_image_flipped = np.fliplr(right_image)
			measurement_flipped = -measurement
			steering_left_flipped = -steering_left
			steering_right_flipped = -steering_right
			images.extend([image_flipped, left_image_flipped, right_image_flipped])
			measurements.extend([measurement_flipped, steering_left_flipped, steering_right_flipped])

X_train = np.array(images)
y_train = np.array(measurements)
print('count:', count)

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((45,20), (0,0))))
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
    history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=3)
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

saved_model = load_trained_model(SAVED_MODEL_PATH)
train_model(saved_model, NEW_MODEL_NAME)
