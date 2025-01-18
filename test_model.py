import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp.hands.Hands(static_image_mode = False, min_detection_confidence = 0.3)

labels_dict = {i:chr(65+i) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H,W,_ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Compute relative positions based on the minimum x and y values
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        mp_drawing.draw_landmarks(
            frame,                         # The video frame or image to draw on
            hand_landmarks,                # Hand landmarks detected by MediaPipe
            mp_hands.HAND_CONNECTIONS,     # Connections between hand landmarks
            mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=4),  # Styling for connections
            mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=2)   # Styling for landmarks
        )

    x1 = int(min(x_) * W) - 10  # Left bound of the bounding box
    y1 = int(min(y_) * H) - 10  # Top bound of the bounding box
    x2 = int(max(x_) * W) + 10  # Right bound of the bounding box
    y2 = int(max(y_) * H) + 10  # Bottom bound of the bounding box

    if len(data_aux) == 42:  # Check if auxiliary data has the expected number of features
        prediction = model.predict([np.asarray(data_aux)])  # Make a prediction using the model
        predicted_char = labels_dict[int(prediction[0])]  # Map the prediction to the corresponding character

        # Draw a rectangle on the video frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  

        # Display the predicted character above the rectangle
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)

cv2.imshow("Sign Detection", frame)  # Show the video frame with overlays
cap.release()  # Release the video capture resource
cv2.destroyAllWindows()  # Close all OpenCV windows
