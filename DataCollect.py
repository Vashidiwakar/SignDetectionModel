import os
import cv2

#making data directory
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

number_of_classes = 26
class_images = 500

cap = cv2.VideoCapture(0) # 0 for default webcam
for j in range (number_of_classes):
    if not os.path.exists(os.path.join(data_dir,str(j))):
        os.makedirs(os.path.join(data_dir,str(j)))

    print('collecting data class {}'.format(j)) # 0->A, 1->B...

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'press enter to capture',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1.3, (0,0,255), 3, cv2.LINE_AA)
        # frame: The image/frame on which the text is to be drawn.
        # 'press enter to capture': The text string to display.
        # (100, 50): The bottom-left corner of the text in the frame (x, y).
        # cv2.FONT_HERSHEY_SIMPLEX: Font type.
        # 1.3: Font scale (size of the text).
        # (0, 255, 0)(B,G,R): Color of the text in BGR format (red in this case).
        # 3: Thickness of the text.
        # cv2.LINE_AA: Line type for anti-aliased text.

        cv2.imshow('frame',frame)
        if cv2.waitKey(25)== 13:
            break 
        # wait for nearly 25ms to consider a key press as a valid key press
        # 13 -> ASCII value of enter (You can give any key (eg. P, then give ascii value of P))
    counter = 0
    while counter < class_images:
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir,str(j),'{}.jpg'.format(counter)), frame)

        counter+=1

        # Allow the user to exit early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

cap.release()
cv2.destroyAllWindows()
        
