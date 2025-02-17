import face_recognition
import os
import pickle
from preprocessing import TRAIN_DIRECTORY, read_names, get_path

def learn():
    known_encodings = []
    known_names = []
    names = read_names()
    i = 0.0
    for person_name in names:
        i+=1.0
        person_path = get_path(TRAIN_DIRECTORY, person_name)
        for img_name in os.listdir(person_path):
            img_path = get_path(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
        print(person_name, i/len(names))
    return known_encodings, known_names

def save(known_encodings, known_names):
    with open("data.pkl", 'wb') as file:
        data = {"encodings": known_encodings, "names": known_names}
        pickle.dump(data, file)


if __name__ == '__main__':
    k_encodings, k_names = learn()
    save(k_encodings, k_names)