import prepare_set as ps
import cv2
from PIL import Image
import numpy as np
import math
import json

TRAINED_FACES_PATH = './trained_faces.json'
TRAIN_PERCENTAGE = 80

def add_to_trained_faces(person_number, name):
    trained_faces = json.load(open(TRAINED_FACES_PATH))
    trained_faces[person_number] = name
    with open(TRAINED_FACES_PATH, 'w') as f:
        json.dump(trained_faces, f, sort_keys=True, indent=4)


def get_person_number():
    trained_faces = json.load(open(TRAINED_FACES_PATH))    
    return len(list(trained_faces.keys()))

def split_train_test(image_list):
    split_value = math.ceil(len(image_list) * TRAIN_PERCENTAGE / 100)
    return image_list[:split_value], image_list[split_value:]

def get_lowest_confidence(predictions, target):
    lowest = 0
    idx = -1
    index = 0
    for prediction, confidence in predictions:
        if (confidence < lowest or lowest == 0) and prediction == target:
            lowest = confidence
            idx = index
        index += 1
    return confidence, idx

def predict(recognizer, prediction_set):
    predictions = []
    for prediction_image in prediction_set:
        predictions.append(recognizer.predict(prediction_image))
    return predictions

def train_for_person(path, recognizer_data=None):
    images, name = ps.segment_into_n_face_list(path)
    images.sort(key=len)
    person_number = get_person_number()
    super_test_set = []
    if len(images[0]) != 1:
        print("Need to handle this case! No 1 person images found")
        return
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained = False

    if recognizer_data is not None:
        face_recognizer.read(recognizer_data)
        trained = True
    current_segment = 1
    faces_of_interest = []

    for face_segment in images:
        if len(face_segment) == current_segment:
            faces_of_interest.append(face_segment)
        else:
            if current_segment == 1:
                flattened_foi = [face[0] for face in faces_of_interest]
                training_set, testing_set = split_train_test(flattened_foi)
                labels = np.array([person_number] * len(training_set))
                super_test_set += testing_set
                if not trained:
                    face_recognizer.train(training_set, labels)
                    trained = True
                else:
                    face_recognizer.update(training_set, labels)
                super_test_set += testing_set
            else:
                foi = []
                for face_segment in faces_of_interest:
                    predictions = predict(face_recognizer, face_segment)
                    conf, idx = get_lowest_confidence(predictions, person_number)
                    foi.append(face_segment[idx])
                training_set, testing_set = split_train_test(foi)
                labels = np.array([person_number] * len(training_set))
                super_test_set += testing_set                
                face_recognizer.update(training_set, labels)
            current_segment = len(face_segment)
            faces_of_interest = []
    add_to_trained_faces(person_number, name)
    return face_recognizer

