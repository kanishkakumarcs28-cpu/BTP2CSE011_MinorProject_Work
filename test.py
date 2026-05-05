import json
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# paths
MODEL_PATH = r"C:\Users\Kanishka Kumar\OneDrive\Desktop\plant-diseases-cnn\plant_disease_model.keras"

img_size = 128

# load model
model = load_model(MODEL_PATH)

# load class names (same order as training)
class_indices = json.load(open("class_indices.json"))
class_names = list(class_indices.keys())

# image from command line
img_path = sys.argv[1]

# preprocess image (same as training)
img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Disease:", predicted_class)
