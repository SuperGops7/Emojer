import cv2 as cv
import numpy as np

from head_classifiers import face_classifier, eye_classifier

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    eyes = eye_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for (x, y, w, h) in eyes:
        cv.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return eyes, faces

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        break
    # Our operations on the frame come here
    
    eyes, faces = detect_bounding_box(frame)
    print(faces)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


