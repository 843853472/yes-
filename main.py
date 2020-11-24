# -*- coding: utf-8 -*-
import api
import cv2
import numpy as np
from keras.models import load_model
import dlib
import time
import argparse
import models
from img import encodings
import time
##################################################
yu_encoding = encodings(1)
hui_face_encoding = encodings(2)
chen_face_encoding = encodings(3)
rui_face_encoding = encodings(4)
mu_face_encoding = encodings(5)
man_face_encoding = encodings(6)
known_face_encodings = [
    yu_encoding,
    hui_face_encoding,
    chen_face_encoding,
    rui_face_encoding,
    mu_face_encoding,
    man_face_encoding,
]
known_face_names = [
    "1 Yu",
    "2 Hui",
    "3 Chen",
    "4 Rui",
    "5 Mu",
    "6 Man",
]
#jian_face_encoding = encodings(7)
#known_face_encodings.append(jian_face_encoding)
#known_face_names.append("7 Jian")
#########################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threshold", type = float, default=0.20,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
	help="the number of consecutive frames the eye must be below the threshold")
args = vars(ap.parse_args())
#########################################################################
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(models.pose_predictor_model_location())
emotion_classifier = load_model(models.simple_CNN_530_0_65_hdf5())
face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
############################################################################
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
emotion=""
trig = 0
delay = 0
ear = args['threshold']
font = cv2.FONT_HERSHEY_DUPLEX
(lStart, lEnd) = api.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = api.FACIAL_LANDMARKS_IDXS["right_eye"]
##########################################################################
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Netural'
}
##########################################################################
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = api.resize(gray, width=450)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = api.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = api.eye_aspect_ratio(leftEye)
            rightEAR = api.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

        if ear < args['threshold']:
            trig = 1
        if trig == 1:
            delay += 1
        if delay >= 6:
            delay = 0
            trig = 0
            face_locations = api.face_locations(rgb_small_frame)
            face_encodings = api.face_encodings(rgb_small_frame, face_locations)
            faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
            for (x, y, w, h) in faces:
                gray_face = gray[(y):(y + h), (x):(x + w)]
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face / 255.0
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                emotion = emotion_labels[emotion_label_arg]
            face_names = []
            for face_encoding in face_encodings:
                matches = api.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = api.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (150, 80), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (10, 30), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, emotion, (10, 60), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)  
    if len(face_locations): 
        time.sleep(3)    
    process_this_frame = not process_this_frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()