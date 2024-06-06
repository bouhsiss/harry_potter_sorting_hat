import pandas as pd
import math

BESSEL_CORRECTION = 1

# Utility function for loading data (assuming it's provided in utils.utils)
from utils.utils import load_data

class DataDescriber:
    def __init__(self, data):
        self.data = data
        self.numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    def _get_min_max(self, data):
        min_val = data[0]
        max_val = data[0]
        for val in data:
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        return min_val, max_val

    def _get_percentiles(self, data):
        data.sort()
        n = len(data) - 1  # 0-indexed
        q1 = data[n // 4]
        q3 = data[3 * n // 4]
        return q1, q3

    def _get_median(self, data):
        data.sort()
        n = len(data) - 1  # 0-indexed
        if n % 2 == 0:
            return (data[n // 2] + data[n // 2 + 1]) / 2
        return data[n // 2]

    def describe(self):
        result = {}
        
        for column in self.numerical_columns:
            col_data = self.data[column].dropna().values  # drop missing values
            count = len(col_data)
            mean = sum(col_data) / count
            variance = sum((x - mean) ** 2 for x in col_data) / (count - BESSEL_CORRECTION)  # sample variance
            std = math.sqrt(variance)
            min_val, max_val = self._get_min_max(col_data)
            q1, q3 = self._get_percentiles(col_data)
            median = self._get_median(col_data)

            result[column] = [
                count,
                mean,
                std,
                min_val,
                q1,
                median,
                q3,
                max_val
            ]

        df_result = pd.DataFrame(result, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        return df_result

# Load the data
data = load_data("data/raw/datasets/dataset_train.csv")

# Create an instance of DataDescriber
describer = DataDescriber(data)

# Get the description
description = describer.describe()
print(description)
print("========================================================")
print(data.describe())
