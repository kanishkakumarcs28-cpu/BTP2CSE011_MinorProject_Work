import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img_size = 128

# model load
model = load_model("model.h5")

# classes (dataset folder se automatically)
import os
class_names = list(model.class_names) if hasattr(model, "class_names") else sorted(os.listdir("dataset"))

# image path command line se lo
img_path = sys.argv[1]

# image load
img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Disease:", predicted_class)
