import os
import pickle

import mediapipe as mp
import cv2

 #3 objects for landmarks-
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)  #Model to detect hand gestures

DATA_DIR = './data'

data = []
labels = [] #stores directory number(for different hand gestures)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []  #LIST OF ALL X-COORDINATES OF LANDMARKS
        y_ = []  #LIST OF ALL Y-COORDINATES OF LANDMARKS

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #for mediapipe

        results = hands.process(img_rgb)  #passes RGB image into our model

        if results.multi_hand_landmarks:  #if we detect at least one hand-
            for hand_landmarks in results.multi_hand_landmarks:

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)   #APPENDING TO THE LIST OF ALL X-COORDINATES OF LANDMARKS
                    y_.append(y)   #APPENDING TO THE LIST OF ALL Y-COORDINATES OF LANDMARKS

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

#saving all the data-
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()