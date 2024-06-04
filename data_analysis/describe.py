import pandas as pd 
from utils.utils import load_data

data = load_data("data/raw/datasets/dataset_train.csv")

# Extracting numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

def get_min_max(data):
    min_val = data[0]
    max_val = data[0]
    for val in data:
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    return min_val, max_val

def get_percentiles(data):
    data.sort()
    n = len(data)
    q1 = data[n // 4]
    q3 = data[3 * n // 4]
    return q1, q3

def get_median(data):
    data.sort()
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2] + data[n // 2 + 1]) / 2
    return data[n // 2]

# Define the describe function
def describe(data, numerical_columns):
    result = {}
    for column in numerical_columns:
        col_data = data[column].dropna().values # drop missing values
        count = len(col_data)
        mean = sum(col_data) / count
        variance = sum((x - mean) ** 2 for x in col_data) / count
        std = variance ** 0.5
        min_val, max_val = get_min_max(col_data)
        q1 ,q3 = get_percentiles(col_data)
        median = get_median(col_data)

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
    
    return result

# Call the describe function
description = describe(data, numerical_columns)

# Print the description
print("================================================= description =================================================")
header = ["Feature"] + list(numerical_columns)
rows = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

print("\t".join(header))
for i, row in enumerate(rows):
    print(row, end="\t")
    for col in numerical_columns:
        print(f"{description[col][i]:.6f}", end="\t")
    print()
print("===============================================================================================================")

# describe using pandas
print("================================================= pandas describe =================================================")
print(data.describe())
print("===============================================================================================================")