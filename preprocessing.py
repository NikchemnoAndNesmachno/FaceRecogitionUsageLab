import os

TRAIN_DIRECTORY = "Faces/"
TEST_DIRECTORY = "TestFaces/"

def get_path(directory: str, name: str):
    return os.path.join(directory, name)

def save_names():
    names = os.listdir(TRAIN_DIRECTORY)
    with open("names.txt", 'w') as file:
        for name in names:
            file.write(name + "\n")

def read_names():
    names = []
    with open("names.txt", 'r') as file:
        for line in file:
            names.append(line.strip())
    return names

def remove_tests(names: list[str]):
    for name in names:
        directory = get_path(TEST_DIRECTORY, name)
        files = os.listdir(directory)
        files = files[:len(files)//2]
        for file in files:
           path = get_path(directory, file)
           os.remove(path)

def remove_trains(names: list[str]):
    for name in names:
        directory = get_path(TRAIN_DIRECTORY, name)
        files = os.listdir(directory)
        files = files[len(files)//2:]
        for file in files:
           path = get_path(directory, file)
           os.remove(path)

if __name__ == '__main__':
    person_names = read_names()
    remove_trains(person_names)
    remove_tests(person_names)
