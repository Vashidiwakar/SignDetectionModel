import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.5)

data_dir = './data'

data = [] # all hand landmarks
labels = [] # corresponding labels (class name)

for dir in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir)
    if not os.path.isdir(dir_path):
        continue  # Skip files, only process directories

    # Iterate through each image in the subdirectory
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        auxdata = []  # Auxiliary data for landmarks
        x = [] #to store x coordinates of landmarks
        y = [] #to store y coordintes of landmarks
        
        #opencv reads images in BGR format but mediapipe needs RGB format
        img = cv2.imread(img_path) #loading image
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image into RGB (if not in RGB)

        results = hands.process(imgrgb) #processing the image with mediapipe hands (mediapipe.Hands)

        #iterating for each hand in every image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)): #for all landmarks(dots in image) created by mediapipe 
                    # Collect normalized x and y coordinates
                    x_ = hand_landmarks.landmark[i].x
                    y_ = hand_landmarks.landmark[i].y

                    # Append to the x and y lists
                    x.append(x_)
                    y.append(y_)

                # After collecting all landmarks, compute relative coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x_ = hand_landmarks.landmark[i].x
                    y_ = hand_landmarks.landmark[i].y

                    # Compute relative positions based on the minimum x and y values
                    auxdata.append(x_ - min(x))
                    auxdata.append(y_ - min(y))
                    
            data.append(auxdata)
            labels.append(dir)


f = open('data.pickle','wb')
pickle.dump({'data':data, 'labels':labels},f) #pickle -> a type of file
f.close()
