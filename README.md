# ASL Gesture Recognition Using MediaPipe and Random Forest

This project focuses on recognizing American Sign Language (ASL) hand gestures from static images using computer vision and machine learning. The core idea is to extract hand landmarks using MediaPipe and then classify the gestures using a Random Forest classifier.

## Project Overview

- Hand landmarks are extracted using MediaPipe from a dataset of static hand gesture images.
- A Random Forest classifier is trained on the extracted landmarks.
- A basic HTML interface is included to display prediction results.

## Folder Structure
asl-gesture-detection-mediapipe/
├── data/
│ └── asl_alphabet_train/ # Folder of training images per class (A-Z)
├── mediapipe_landmark_extractor.py # Extracts hand landmarks into a CSV file
├── rf_model.ipynb # Jupyter notebook to train Random Forest
├── app/
│ ├── index.html # Frontend to display predictions
│ └── hand.csv # Prediction input
├── model.pkl # Trained model (optional - not pushed)
└── README.md

## How to Run

1. Install the required Python packages:
2. Run the landmark extraction script:
3. Train the classifier:
4. To test your model, you can open the HTML file:

## Key Technologies Used

- MediaPipe for extracting hand landmarks from images
- Random Forest from scikit-learn for classification
- HTML and optional JavaScript for the user interface

## Author

Vashi Diwakar  
B.Tech Electrical Engineering  
Indian Institute of Technology Kanpur
