import os
from PIL import Image
import numpy as np
import cv2

haar_cascade = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade)


def segment_into_n_face_list(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    name = folder_path[folder_path.rfind('/') + 1:]
    face_segments = []
    for image_path in image_paths:
        images_in_photo = []
        grayscale = Image.open(image_path).convert("L")
        image = np.array(grayscale, "uint8")
        faces = face_cascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images_in_photo.append(image[y: y + h, x: x + w])
        if len(images_in_photo) > 0:
            face_segments.append(images_in_photo)
    return face_segments, name
