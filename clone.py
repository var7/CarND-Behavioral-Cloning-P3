import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

MODEL = 'newbestdata1'
lines = []

FOLDER = './newbestdata/'
IMGPATH = FOLDER + 'IMG/'
with open(FOLDER+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

correction = 0.2
images = []
measurements = []
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
		images.extend([image, left_image, right_image])
		measurement = float(line[3])
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation = "elu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Dropout(0.25))
model.add(Conv2D(48,5,5,subsample=(2,2), activation='elu'))
model.add(Conv2D(64,3,3,activation="elu"))
model.add(Conv2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=5)

model.save(MODEL+'.h5')

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(MODEL+'.png', bbox_inches='tight')
exit()
