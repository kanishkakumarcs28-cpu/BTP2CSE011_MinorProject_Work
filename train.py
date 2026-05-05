import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
from tensorflow.keras import layers, models
DATASET_PATH = r"C:\Users\Kanishka Kumar\OneDrive\Desktop\plant-diseases-cnn\Dataset"

img_size = 128
batch = 32

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch,
    subset="training",
)
json.dump(train.class_indices, open("class_indices.json", "w"))

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch,
    subset="validation",
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train, validation_data=val, epochs=25)

import pickle
pickle.dump(history.history, open("history.pkl", "wb"))

model.save("plant_disease_model.keras")
