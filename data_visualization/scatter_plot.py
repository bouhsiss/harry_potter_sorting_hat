import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from utils.utils import load_data

# creating a scatter plot answering the question:
# "what are the two features that are similar ?"

# Load the data
data = load_data("data/raw/datasets/dataset_train.csv")

# List of courses to plot
courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

# Generate all combinations of course pairs
course_pairs = list(combinations(courses, 2))

# Set up the plotting
num_plots = len(course_pairs)
cols = 8
rows = (num_plots // cols) + (num_plots % cols > 0)

plt.figure(figsize=(20, 4 * rows))

# Loop through each combination and create scatter plots
for i, (course1, course2) in enumerate(course_pairs):
    plt.subplot(rows, cols, i + 1)
    scatter = sns.scatterplot(x=data[course1], y=data[course2])
    plt.xlabel(course1)
    plt.ylabel(course2)

plt.tight_layout()
plt.show()