# 🌿 Plant Leaf Diseases Detection Using CNN

This project is a Deep Learning-based solution to identify various plant diseases from leaf images. Using **Transfer Learning** with the **MobileNetV2** architecture, the model achieves high accuracy while remaining lightweight enough for mobile applications.

## 🚀 Key Features
*   **Deep Learning Model:** Built with TensorFlow/Keras using MobileNetV2.
*   **Dataset:** Utilizes the PlantVillage dataset with 38 distinct classes of healthy and diseased leaves.
*   **Interactive Notebooks:** Step-by-step training and evaluation process.
*   **Visualization:** Includes training history curves and normalized confusion matrices.

## 📂 Project Structure
*   `Train_plant_disease.ipynb`: Contains the full training pipeline and accuracy/loss visualizations.
*   `Test_plant_disease.ipynb`: Model evaluation script featuring the Confusion Matrix and single image prediction.
*   `plant_disease_model.keras`: The final trained model weights.
*   `class_indices.json`: Mapping of class indices to disease names.

## 📊 Results

### Training Performance
The model shows steady improvement over 25 epochs.
![Training Curves](training_curves.png)

### Confusion Matrix
The normalized confusion matrix below demonstrates the model's precision across all categories.
![Confusion Matrix](normalized_confusion_matrix.png)

## 🛠️ Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install tensorflow numpy matplotlib seaborn scikit-learn`.
3. Open `Test_plant_disease.ipynb` to run predictions on your own leaf images.

## 👤 Author
**Kanishka kumar**
