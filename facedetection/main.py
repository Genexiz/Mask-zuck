import cv2
import tensorflow
import keras
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np





face_cascade = "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(face_cascade)
webcam = cv2.VideoCapture(0)
np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
size = (224, 224)
count = 0
while True:

    success,img_bgr = webcam.read()
    image_org = img_bgr.copy()
    image_bw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    faces = face_classifier.detectMultiScale(image_bw)



    for face in faces:

        x,y,w,h = face
        cface_rgb = Image.fromarray(image_rgb[y:y + h, x:x + w])

        cv2.imwrite(f'./facedetection/test/person_{count}.jpg', image_org[y:y + h, x:x + w])
        count += 1




        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = cface_rgb

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        cv2.rectangle(img_bgr, (x, y), (x + h, y + h), (0, 255, 0), 2)
        if prediction[0][0] > prediction[0][1]:

            cv2.putText(img_bgr, 'Non-Masked', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(img_bgr, (x, y), (x + h, y + h), (0, 0, 255), 2)
            # save_dir = './facedetection/Test_save/'
            # filename = f'person_{count}.jpg'
            # save_path = save_dir + filename
            # cv2.imwrite(save_path, image_org[y:y + h, x:x + w])
            # count += 1
        else:
            cv2.putText(img_bgr, 'Masked', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(img_bgr, (x, y), (x + h, y + h), (0, 255, 0), 2)





    cv2.imshow("Mask Detection", img_bgr)
    cv2.waitKey(1)
