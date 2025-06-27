import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming dataset/ is structured with folders for each sign label (A-Z or custom)
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = data_gen.flow_from_directory('dataset', target_size=(64, 64), color_mode='grayscale',
                                          class_mode='categorical', subset='training')
val_data = data_gen.flow_from_directory('dataset', target_size=(64, 64), color_mode='grayscale',
                                        class_mode='categorical', subset='validation')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

model.save('model/gesture_model.h5')
