import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data

# Load the data
data = load_data("data/raw/datasets/dataset_train.csv")

# List of courses to plot
courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

# Set up the matplotlib figure
plt.figure(figsize=(15,10))

# Create box plots for each course
for i, course in enumerate(courses, 1):
    plt.subplot(4, 4, i)
    for house in data["Hogwarts House"].unique():
        subset = data[data["Hogwarts House"] == house]
        sns.histplot(subset[course], kde=False, bins=20, label= house, alpha=0.6)
    plt.title(f'{course} Scores by Hogwarts House')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()    