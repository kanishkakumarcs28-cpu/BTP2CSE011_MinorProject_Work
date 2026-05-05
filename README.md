<h1 align="center">🌿 Plant Disease Detection using CNN</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Model-MobileNetV2-blue?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Accuracy-99.32%25-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow">
</p>

---

## 📊 Project Overview & Results
This project uses **Transfer Learning (MobileNetV2)** to classify 38 different classes of plant leaf diseases. It includes a complete pipeline from training to web deployment.

### 📈 Training Performance
The model was trained for 25 epochs. Below are the Accuracy and Loss curves:
![Training Curves](<img width="3000" height="2100" alt="training_curves" src="https://github.com/user-attachments/assets/a2c999e0-faf4-4073-871b-1f6ba5180ad6" />
)

### 🧩 Confusion Matrix
To evaluate precision across all 38 classes, a normalized confusion matrix was generated:
![Confusion Matrix](<img width="4200" height="3600" alt="normalized_confusion_matrix" src="https://github.com/user-attachments/assets/516e07c1-1699-4d9c-8f4e-d3c7e520f8dd" />
)

### 💻 Web Interface
I developed a functional web application to provide real-time predictions with confidence scores, making the model accessible for practical use.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c90281a-8a02-45e1-ae43-3263099504cd" width="600" alt="Plant Disease Web App">
</p>

---

## 🛠️ Tech Stack Used in This Project
<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

---

## 📂 Repository Structure
- 📁 `Train_plant_disease.ipynb`: Model architecture and training logic.
- 📁 `Test_plant_disease.ipynb`: Confusion matrix and performance evaluation.
- 📁 `app.py`: Backend script for the Web Application.
- 📁 `templates/ & static/`: Frontend files for the web UI.
- 📄 `plant_disease_model.keras`: Final trained model weights.

---

## 🚀 About the Developer
<h3 align="left">Hi 👋, I'm Kanishk Kumar</h3>
- 🎓 **B.Tech CSE Student** at IILM University, Greater Noida.
- 💡 **Specialization:** AI Enthusiast & Passionate Developer.
- 📈 **Goal:** To build AI-driven solutions for real-world impact.

<p align="left">
  <img src="https://github-readme-stats.vercel.app/api?username=kanishkakumarcs28-cpu&show_icons=true&theme=tokyonight&height=150"/>
</p>

---

⭐ **Thank you for visiting!** If you find this project helpful, feel free to star the repository.
