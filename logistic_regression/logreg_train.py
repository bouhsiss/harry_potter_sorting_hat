import numpy as np
import pickle
from data_preprocessing.data_preprocessing import DataPreprocessing
import json
from sklearn.metrics import accuracy_score
import sys


# a multi-class logistic regression model class
class LogisticRegressionOvR:
    def __init__(self, learning_rate=0.1, n_iterations= 5000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.classes = None
        self.decoded_classes = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_weights(self, n_features, n_classes):
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def create_mini_batches(self, X, y):
        m = X.shape[0]
        mini_batches = []

        # shuffle the data
        indices = np.arange(m)
        np.random.shuffle(indices)

        X_shuffled = X.iloc[indices]
        y_shuffled = y[indices]

        # create mini-batches
        for i in range(0, m, self.batch_size):
            X_batch = X_shuffled[i:i+self.batch_size]
            y_batch = y_shuffled[i:i+self.batch_size]
            mini_batches.append((X_batch, y_batch))
        
        return mini_batches


    def mini_batch_gradient_descent(self, X, y, class_idx):
        mini_batches = self.create_mini_batches(X, y)

        for X_batch, y_batch in mini_batches:

            # compute the predictions
            z = np.dot(X_batch, self.weights[class_idx]) + self.bias[class_idx]
            y_pred = self.sigmoid(z)

            # compute the gradients
            m = X_batch.shape[0]
            dw = (1/m) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1/m) * np.sum(y_pred - y_batch)

            # update the weights and bias
            self.weights[class_idx] -= self.learning_rate * dw
            self.bias[class_idx] -= self.learning_rate * db
 
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.initialize_weights(n_features, n_classes)

        # train one classifier per class using the One-vs-Rest strategy
        for class_idx in range(n_classes):
            y_class = np.where(y == self.classes[class_idx], 1, 0)

            for _ in range(self.n_iterations):
                self.mini_batch_gradient_descent(X, y_class, class_idx)

    
    def save_weights(self, weights_file):
        """Save the model weights and biases to a file"""
        model_data = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "classes": self.decoded_classes.tolist()
        }

        with open(weights_file, "w") as file:
            json.dump(model_data, file)

        

def main():
    # check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_file>")
        sys.exit(1)
    # load the training data and preprocess it
    train_file = sys.argv[1]
    weights_file = "logistic_regression_weights.txt"
    preprocessor = DataPreprocessing(train_file) 
    preprocessor.preprocess()
    # split the data into training and testing sets
    X_train, _, y_train, _ = preprocessor.split_data()

    # train the logistic regression model
    model = LogisticRegressionOvR()
    model.fit(X_train, y_train)
    
    # save the model weights, biases and classes to a file
    model.decoded_classes = preprocessor.decode_categorical_data(model.classes)
    model.save_weights(weights_file)




if __name__ == "__main__":
    main()