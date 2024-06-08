import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data

# Load the data
data = load_data("data/raw/datasets/dataset_train.csv")

# List of courses to plot
courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

# Set up the matplotlib figure
plt.figure(figsize=(15,10))

# define colors for each house
house_colors = {
    "Ravenclaw": "#0e1a40",
    "Slytherin": "#1a472a",
    "Gryffindor": "#740001",
    "Hufflepuff": "#ecb939"

}

# Create box plots for each course
for i, course in enumerate(courses, 1):
    plt.subplot(4, 4, i)
    data_list = [data[data["Hogwarts House"] == house][course].dropna() for house in data["Hogwarts House"].unique()]
    colors = [house_colors[house] for house in data["Hogwarts House"].unique()]
    plt.hist(data_list, bins=20, stacked=True, label=data["Hogwarts House"].unique(), color=colors, alpha=0.7)
    plt.title(f'{course} Scores by Hogwarts House')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()

# interpretation
# to identify the most homogenous distribution score we should take in consideration these factors among all four houses:
# shape consistency (the shape is consistent across all houses) - range and spread is similar across the houses
# central tedency (mean and median) the more these values are similar across all houses the more homogenous the distribution
# overlapping areas (more overlap indicates more similarity in distributions)

# after plotting our data we observe that there are two courses that checks these criterias : 
# -- Arithmancy and care of magical creatures -- 
# but there are some differences, care of magical creatures shows more homogeneity. the narrow range, aligned peaks
# and high degree of overlap suggest that the score distributions are more similar across the houses compared to arithmancy which has a wider range and spread and some outliers 