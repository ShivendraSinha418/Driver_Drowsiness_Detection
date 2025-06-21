# 💤 Drowsy Driver Classifier

A deep learning model that detects driver drowsiness from face images using a custom-built Convolutional Neural Network (CNN).  
The project is implemented entirely in a Jupyter notebook and aims to support real-time driver monitoring systems.

---

![Banner](https://github.com/ShivendraSinha418/Driver_Drowsiness_Detection/blob/main/dw.png)

---

## 🧠 Overview

Driver fatigue is a major cause of road accidents. This project presents a **CNN-based solution** that classifies whether a driver is **drowsy** or **alert** based on facial images. The model is trained on a labeled dataset consisting of drowsy and non-drowsy face images.

---

## 🚀 Features

- Custom CNN model built from scratch using Keras
- Binary classification: **Drowsy** vs **Normal**
- Trained on a custom dataset of face images
- Implemented entirely in a Jupyter notebook
- Suitable for integration with real-time camera input (future work)

---

## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- OpenCV (for face image preprocessing)  
- NumPy, Pandas  
- Jupyter Notebook

---
## 🧪 Model Architecture (Custom CNN)

The model uses a custom deep CNN architecture designed for binary classification (drowsy vs alert). It includes multiple convolutional layers, batch normalization for stability, and dense layers with dropout to reduce overfitting.

python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

