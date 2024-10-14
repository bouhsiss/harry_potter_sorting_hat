from sklearn.metrics import accuracy_score
from data_preprocessing.data_preprocessing import DataPreprocessing
from logistic_regression.logreg_predict import load_weights, predict
import sys



def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {accuracy}")

def main():
    # check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python logreg_train.py <dataset_file> <weights_file>")
        sys.exit(1)
    # load the training data and preprocess it
    train_file = sys.argv[1]
    preprocessor = DataPreprocessing(train_file) 
    preprocessor.preprocess()
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = preprocessor.split_data()

    # load the weights file and predict
    weights_file = sys.argv[2]
    weights, biases, classes = load_weights(weights_file)

    y_pred = predict(X_test, weights, biases, classes)

    # replace the y_pred with the decoded classes
    y_test = classes[y_test]

    # evaluate the model
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    main()