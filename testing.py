import pickle
import face_recognition
import os
import numpy as np
import pandas as pd
from preprocessing import TEST_DIRECTORY, read_names, get_path


def recognize_face(image, known_encodings, known_names, threshold=0.52):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    recognized_faces = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = min(distances)

        if min_distance < threshold:
            index = np.argmin(distances)
            recognized_faces.append(known_names[index])
        else:
            recognized_faces.append("Unknown")
    return recognized_faces


def test(known_encodings, known_names):
    results = []
    names = read_names()
    i = 0.0
    for person_name in names:
        i += 1.0
        person_path = get_path(TEST_DIRECTORY, person_name)
        for image in os.listdir(person_path):
            image_path = get_path(person_path, image)
            test_image = face_recognition.load_image_file(image_path)
            recognized_faces = recognize_face(test_image, known_encodings, known_names, threshold=0.6)
            results.append((person_name, recognized_faces))
        print(person_name, int(i/len(names) * 100))
    results_df = pd.DataFrame(results, columns=["Image", "Recognized Faces"])
    return results_df

def save_test_results(dataframe):
    with open("test_results.pkl", 'wb') as file:
        pickle.dump(dataframe, file)

def read():
    with open("data.pkl", 'rb') as file:
        data = pickle.load(file)
        return data["encodings"], data["names"]

if __name__ == '__main__':
    e, k = read()
    result = test(e, k)
    save_test_results(result)