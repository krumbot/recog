import prepare_set as ps
import cv2
from PIL import Image
import numpy as np
import math

TRAIN_PERCENTAGE = 80

def get_person_number():
    return 0

def split_train_test(image_list):
    split_value = math.ceil(len(image_list) * TRAIN_PERCENTAGE / 100)
    return image_list[:split_value], image_list[split_value:]

def get_lowest_confidence(predictions):
    lowest = 0
    idx = -1
    for index, prediction, confidence in enumerate(predictions):
        if confidence < lowest or if lowest == 0:
            lowest = confidence
            idx = index
    return confidence, idx

def predict(recognizer, prediction_set):
    predictions = []
    for prediction_image in prediction_set:
        predictions.append(recognizer.predict(prediction_image))
    return predictions

def train_for_person(path, prev_recognizer=None):
    images, name = ps.segment_into_n_face_list(path)
    images.sort(key=len)
    person_number = get_person_number()
    super_test_set = []
    if len(images[0]) != 1:
        print("Need to handle this case! No 1 person images found")
        return
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained = False

    if prev_recognizer is not None:
        face_recognizer.read(prev_recognizer)
        trained = True
    current_segment = 1
    faces_of_interest = []

    for face_segment in images:
        if len(face_segment) == current_segment:
            faces_of_interest += face_segment
        else:
            if current_segment == 1:
                training_set, testing_set = split_train_test(faces_of_interest)
                labels = np.array([person_number] * len(training_set))
                if not trained:
                    face_recognizer.train(training_set, labels)
                    trained = True
                else:
                    face_recognizer.update(training_set, labels)
                super_test_set += testing_set
            else:
                predictions = predict(face_recognizer, faces_of_interest)
                conf, idx = get_lowest_confidence(predictions)


            current_segment = len(face_segment)
            faces_of_interest = []
    return face_recognizer


