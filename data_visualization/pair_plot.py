import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data

# creating a pair plot that will answer the question:
# "from the visualization, what features are we going to use for the logistic regression model ?"


# Load the data
data = load_data("data/raw/datasets/dataset_train.csv")

# List of courses to plot
courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]


# define colors for each house
house_colors = {
    "Ravenclaw": "#0e1a40",
    "Slytherin": "#1a472a",
    "Gryffindor": "#740001",
    "Hufflepuff": "#ecb939"

}


# Create the pair plot
sns.pairplot(data, vars=courses, hue='Hogwarts House', palette=house_colors, diag_kind='kde')

plt.show()