import pandas as pd
from utils.utils import load_data


data  = load_data("data/raw/datasets/dataset_train.csv")


# print the first 5 rows of the dataset
print("================================================= data head =================================================")
print(data.head())

## ======== interpretation ======== ## 
# the dataset contains informations about students, including their hogwarts house, first name, last name, birthday, best hand and scores in various subjects.
# there's already a column named index, which is the same as the index of the dataframe, TO DO : so we can drop it.
# the 'hogwarts house' column is categorical and represents the house to which the student belongs. this is the target column that we want to predict.
# the 'first name and 'last name' columns are categorical and might not be useful for a the ML model but can be used to identify the students. TO DO : we can drop them.
# the 'best hand' column is categorical and represents the hand that the student uses the most. this column might be useful for the ML model.
# some score columns have negative values.

# count how many negative values the dataframe has to know how they should be treated
print("================================================= negative values =================================================")
numerical_cols = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
negative_values_count = (data[numerical_cols] < 0).sum()
print(negative_values_count)
total_negative_values = negative_values_count.sum()
print(f"total negative values: {total_negative_values}")

# Given that the number of negative values is very highm, especially in the Charms column since all its values are negative, we can assume that these negative values are not errors but rather have a specific meaning.



# print the last 5 rows of the dataset
print("================================================= data tail =================================================")
print(data.tail())

## ======== interpretation ======== ##
# the dataset contains 1600 rows and 19 columns including the index column and target column.
# the dataset seems to be sorted by the index column, which is the same as the index of the dataframe.


# print data types of the columns
print("================================================= data types =================================================")
print(data.dtypes)

## ======== interpretation ======== ##
# index: int64 - the unique identifier of the student
# hogwarts house - first name - last name - best hand - birthday: object (mostly a string) - categorical columns 
# the rest of the columns ==> (arithmancy - astronomy - herbology - defense against the dark arts - divination - muggle studies - ancient runes - history of magic - transfiguration - potions - care of magical creatures - charms - flying) : float64 - numerical columns


# print the number of missing values in each column
print("================================================= missing values =================================================")
print(data.isnull().sum())


## ======== interpretation ======== ##
# the columns (index - hogwarts house - first name - last name - birthday - best hands - charms - flying) have no missing values.
# the rest of the columns have missing values that ranges from 30 to 40 missing values. TO DO : we need to decide how to handle these missing values.



# value counts for the categorical columns

print("================================================= hogwarts house  value counts =================================================")
print(data["Hogwarts House"].value_counts())

## output ##
# Hufflepuff    529
# Ravenclaw     443
# Gryffindor    327
# Slytherin     301
## ======== interpretation ======== ##
# hufflepuff has the most of students and slytherin has the least of students.

print("================================================= best hand  value counts =================================================")
print(data["Best Hand"].value_counts())

## output ##
# Right    810
# Left     790
## ======== interpretation ======== ##
# the students are almost equally distributed between right and left handed students.


# print the cross tabulation of the best hand and hogwarts house
print(pd.crosstab(data["Best Hand"], data["Hogwarts House"], normalize="index"))

## output ##
#                 Gryffindor  Hufflepuff  Ravenclaw  Slytherin
# Left              0.212658    0.332911   0.269620   0.184810
# Right             0.196296    0.328395   0.283951   0.191358
## ======== interpretation ======== ##
# The distributions of the best hand across the hogwarts houses does seem to follow the overall distribution of the students across the houses.
# this suggests that the best hand may not have a strong relationship with the hogwarts house feature.