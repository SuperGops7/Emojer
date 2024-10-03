import cv2 as cv

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_eye.xml"
)