import pickle
import pandas as pd

def read():
    with open("test_results.pkl", 'rb') as file:
        return pickle.load(file)

def evaluate(results: pd.DataFrame):
    right = 0.0
    count = 0.0
    accuracies = {}
    rows = results.iterrows()
    for index, row in rows:
        true = row["Image"]
        if true not in accuracies.keys():
            accuracies[true] = [0.0, 0.0]
        predicted = row["Recognized Faces"]
        if len(predicted) != 0:
            if predicted[0] == true:
                accuracies[true][0] += 1
                right += 1
        accuracies[true][1] += 1
        count+=1

    for key in accuracies:
        values = accuracies[key]
        print(f"{key}: ", values[0]/values[1])
    print(f"\t Total accuracy: {right/count * 100:.2f}%")


if __name__ == '__main__':
    data = read()
    print(data.to_string())
    evaluate(data)