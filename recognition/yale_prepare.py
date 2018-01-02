import cv2
import os
import numpy as np
from PIL import Image

haar_cascade = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def fetch_image_paths(path, training):
    def fetch_training(f):
        if training:
            return not f.endswith('.sad')
        return f.endswith('.sad')

    return [os.path.join(path, f) for f in os.listdir(path) if fetch_training(f) and f.startswith('subject')]


def get_images_and_labels(path):
    image_paths = fetch_image_paths(path, True)
    images = []
    labels = []
    for image_path in image_paths:
        grayscale = Image.open(image_path).convert("L")
        image = np.array(grayscale, "uint8")
        label_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = face_cascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label_number)
    return images, labels


def predict_faces(path, recognizer):
    image_paths = fetch_image_paths(path, False)
    for image_path in image_paths:
        prediction_image_grayscale = Image.open(image_path).convert("L")
        prediction_image = np.array(prediction_image_grayscale, 'uint8')
        faces = face_cascade.detectMultiScale(prediction_image)
        for (x, y, w, h) in faces:
            prediction, confidence = recognizer.predict(prediction_image[y: y + h, x: x + w])
            actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            if actual == prediction:
                print("{} is Correctly recognized with confidence {}".format(actual, confidence))
            else:
                print("{} is Incorrectly recognized as {} with confidence of {}").format(actual, prediction, confidence)


faces_path = "assets/yale/"
images, labels = get_images_and_labels(os.path.join(os.path.dirname(__file__), faces_path))
print(images[0])
face_recognizer.train(images, np.array(labels))

predict_faces(faces_path, face_recognizer)

cv2.destroyAllWindows()
