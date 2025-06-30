# ASL Gesture Recognition Using MediaPipe and Random Forest

This project focuses on recognizing American Sign Language (ASL) hand gestures from static images using computer vision and machine learning. The core idea is to extract hand landmarks using MediaPipe and then classify the gestures using a Random Forest classifier.

## Project Overview

- Hand landmarks are extracted using MediaPipe from a dataset of static hand gesture images.
- A Random Forest classifier is trained on the extracted landmarks.
- A basic HTML interface is included to display prediction results.

## Project Structure

- `app.py` — Main Flask app for running the web server
- `index.html` — Frontend UI for prediction
- `requirements.txt` — Python package dependencies
- `start.sh` — Shell script to start the app
- `data/` — Folder to store collected data
- `data.pickle`, `scaler.pkl`, `label_encoder.pkl` — Saved preprocessed objects
- `Model_training.py` — Trains the ML model
- `Model_testing.py` — Evaluates the model
- `random_forest_model.pkl` — Final trained model
- `Landmark_Creation.py`, `Data_collection.py` — Scripts to extract hand landmarks using MediaPipe
- `.venv/` — Virtual environment
  
## How to Run

1. Install the required Python packages
2. Run the landmark extraction script
3. Train the classifier
4. To test your model, you can open the HTML file

## Key Technologies Used

- MediaPipe for extracting hand landmarks from images
- Random Forest from scikit-learn for classification
- HTML for the user interface

## Author

Vashi Diwakar  
B.Tech Electrical Engineering  
Indian Institute of Technology Kanpur
