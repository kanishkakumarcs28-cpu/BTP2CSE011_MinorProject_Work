import os
import numpy as np
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("plant_disease_model.keras")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

IMG_SIZE = 128

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    top_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[top_index]
    formatted_class = predicted_class.replace("___", " - ").replace("_", " ")
    confidence = float(prediction[0][top_index])
    return formatted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_image = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction, confidence = predict_disease(filepath)
            uploaded_image = "/" + filepath.replace("\\", "/")

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        uploaded_image=uploaded_image
    )

if __name__ == "__main__":
    app.run(debug=True)
