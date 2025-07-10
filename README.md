# 🤖 **American Sign Language (ASL) Recognition Using MediaPipe and Random Forest** ✋

## Overview
This project leverages **MediaPipe** for hand gesture detection and **Random Forest** for classifying hand gestures into predefined classes. The model processes real-time hand gestures captured through a webcam, predicts the corresponding gesture, and displays the result. It is designed to recognize up to **17 different hand gestures** (A-Z).

## 🛠️ **Technologies Used**
- **OpenCV** – For video capture and image processing 
- **MediaPipe** – For hand landmark detection 
- **Scikit-learn** – For Random Forest classification model 
- **Pickle** – For saving and loading the trained model 
- **NumPy** – For data manipulation 
- **Matplotlib** – For visualization of results 

## ✨ **Features**
- **Dataset Collection**: Collects images of hand gestures and stores them for model training 
- **Data Preprocessing**: Normalizes hand landmarks data for training 
- **Model Training**: Trains a Random Forest classifier on hand gesture data 
- **Gesture Prediction**: Real-time hand gesture recognition using webcam input 
- **Performance Evaluation**: Model evaluation with accuracy score 

## 📈 **Results**
- **Accuracy**: The trained Random Forest model achieved high accuracy in gesture classification 
- **Real-Time Inference**: Real-time hand gesture detection with webcam input 
- **Visualization**: Displays predicted gesture on the screen with bounding boxes around the hand 

## 💻 **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Hand-Gesture-Recognition.git
   cd Hand-Gesture-Recognition
   ```

2. **Install dependencies**:
   You need to install the following libraries to run the project:
   ```bash
   pip install opencv-python mediapipe scikit-learn matplotlib numpy
   ```

## 📝 **Usage Instructions**

1. **Collect Images for Dataset**:
   Run the `collect_imgs.py` script to collect images for training your model. The script will capture images of hand gestures through your webcam and save them in the dataset folder.

   ```bash
   python collect_imgs.py
   ```

2. **Prepare Dataset for Training**:
   Once the images are collected, preprocess them using the `create_dataset.py` script to extract the hand landmarks and save them in a pickle file (`data.pickle`).

   ```bash
   python create_dataset.py
   ```

3. **Train Classifier**:
   Run the `train_classifier.py` script to train the **Random Forest classifier** on the collected dataset. This will generate a trained model saved as `model.p`.

   ```bash
   python train_classifier.py
   ```

4. **Inference (Real-Time Gesture Recognition)**:
   To use the trained model for real-time gesture prediction, run the `inference_classifier.py` script. The webcam will display the recognized gesture in real-time.

   ```bash
   python inference_classifier.py
   ```

## 🔍 **Model Evaluation**
- **Accuracy**: The accuracy of the Random Forest model is printed after training, reflecting the classification performance on the test set.

## 📂 **Files**

- **collect_imgs.py**: Script to collect hand gesture images using a webcam 
- **create_dataset.py**: Preprocesses the collected images and extracts hand landmarks 
- **train_classifier.py**: Trains a Random Forest classifier using the processed data 
- **inference_classifier.py**: Uses the trained model to predict hand gestures in real-time 
- **model.p**: The saved trained model 
- **data.pickle**: Pickled file containing the training data and corresponding labels 

## 🤝 **Contributing**
Feel free to fork this repository and submit pull requests if you have improvements or suggestions! 

## 📄 **License**
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
