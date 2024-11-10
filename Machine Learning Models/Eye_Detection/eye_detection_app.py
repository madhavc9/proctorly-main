# IMPORTING NECESSARY LIBRARIES
import cv2
import numpy as np
import dlib

# Loading Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Capturing live video stream from webcam
cap = cv2.VideoCapture(0)

def detect(gray, frame):
    # Detecting faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detecting eyes in the face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

while True:
    # Reading frame from the video capture
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)  # Detecting the features

    # Display the resulting frame
    cv2.imshow('Eye Detection', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv2.destroyAllWindows()