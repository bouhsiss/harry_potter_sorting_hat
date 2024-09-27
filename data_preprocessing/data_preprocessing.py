import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, file_path):
        """
        Initializes the class with the dataset file path and loads the data.
        """
        self.data = pd.read_csv(file_path)
        self.numerical_cols = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
        self.categorical_cols = ["Best Hand"]
        self.target_col = "Hogwarts House"
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()

    def drop_columns(self, columns):
        """
        Drops the specified columns from the dataset.
        """
        self.data = self.data.drop(columns=columns)

    def fill_missing_values(self):
        """
        Fills missing values in numerical columns with the median of the column values.
        """
        self.data[self.numerical_cols] = self.data[self.numerical_cols].fillna(self.data[self.numerical_cols].median())
    
    def scale_numerical_data(self):
        """
        Scales the numerical columns using MinMaxScaler.
        """
        self.data[self.numerical_cols] = self.scaler.fit_transform(self.data[self.numerical_cols])

    def encode_categorical_data(self):
        """
        Encodes the categorical columns using LabelEncoder.
        """
        self.data["Best Hand"] = self.data["Best Hand"].apply(lambda x: 1 if x == "Right" else 0)
        self.data["Hogwarts House"] = self.le.fit_transform(self.data["Hogwarts House"])
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def preprocess(self):
        """
        Main method to preprocess the data.
        """
        # drop non-necessary columns
        self.drop_columns(["Index", "First Name", "Last Name"])

        # fill missing values
        self.fill_missing_values()

        # scale numerical data
        self.scale_numerical_data()

        # encode categorical data
        self.encode_categorical_data()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test