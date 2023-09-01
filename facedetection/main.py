import cv2
import tensorflow
import keras
from PIL import Image
face_cascade = "haarcascade_frontalface_default.xml"

webcam = cv2.VideoCapture(0)
count = 0
while True:

    success,img_bgr = webcam.read()
    image_org = img_bgr.copy()
    image_bw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(image_bw)

    print(f'There are {len(faces)} face found.')

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(img_bgr,(x,y),(x+h,y+h),(0,255,0),2)
        cv2.imwrite(f'D:/non-mask/non_mask_{count}.jpg', image_org[y:y+h,x:x+w])
        count += 1



    cv2.imshow("Face found", img_bgr)
    cv2.waitKey(1)
