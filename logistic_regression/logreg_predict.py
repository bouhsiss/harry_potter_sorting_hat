import numpy as np
import pandas as pd
import sys
import json
from data_preprocessing.data_preprocessing import DataPreprocessing


def load_weights(weights_file):
    with open(weights_file, "r") as file:
        model_data = json.load(file)
    
    weights = np.array(model_data["weights"])
    biases = np.array(model_data["bias"])
    classes = np.array(model_data["classes"])
    
    return weights, biases, classes

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, biases, classes):
    n_samples = X.shape[0]
    n_classes = len(classes)
    y_pred = np.zeros((n_samples, n_classes))

    for class_idx in range(n_classes):
        z = np.dot(X, weights[class_idx]) + biases[class_idx]
        y_pred[:, class_idx] = sigmoid(z)

    y_pred = np.argmax(y_pred, axis=1)

    # replace the y_pred with the decoded classes
    y_pred = classes[y_pred]

    # convert y_pred to a dataframe
    y_pred = pd.DataFrame(y_pred, columns=["Hogwarts House"])
    y_pred.to_csv("houses.csv")
    return y_pred

def main():
    # check if the correct number of arguments is provided 
    if(len(sys.argv) != 3):
        print("Usage : python logreg__predict.py <test_file> <weights_file>")
        sys.exit(1)
    
    # load the weights file
    weights_file = sys.argv[2]
    weights, biases, classes = load_weights(weights_file)
    
    # load the test file and the scaler and preprocess the data
    test_file = sys.argv[1]
    preprocessor = DataPreprocessing(test_file)
    preprocessor.load_scaler("scaler.pkl")
    test_data = preprocessor.preprocess()
    test_data = test_data.drop(columns=["Hogwarts House"])

    # run the predictions
    predict(test_data, weights, biases, classes)


if __name__ == "__main__":
    main() 