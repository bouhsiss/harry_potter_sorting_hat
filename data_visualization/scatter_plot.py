import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from utils.utils import load_data
import sys


def scatter_plot(data):
    # List of courses to plot
    courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

    # Generate all combinations of course pairs
    course_pairs = list(combinations(courses, 2))

    # Set up the plotting
    num_plots = len(course_pairs)
    cols = 12
    rows = (num_plots // cols) + (num_plots % cols > 0)

    plt.figure(figsize=(3*cols, 4 * rows))

    # define colors for each house
    house_colors = {
        "Ravenclaw": "#0e1a40",
        "Slytherin": "#1a472a",
        "Gryffindor": "#740001",
        "Hufflepuff": "#ecb939"

    }

    # Loop through each combination and create scatter plots
    for i, (course1, course2) in enumerate(course_pairs):
        plt.subplot(rows, cols, i + 1)
        sns.scatterplot(x=data[course1], y=data[course2], hue=data['Hogwarts House'], palette=house_colors, legend=False)
        plt.xlabel(course1)
        plt.ylabel(course2)
        plt.tight_layout()

    plt.tight_layout()
    plt.show()


# interpretation
# scatter plots primary use are to observe and show the relationship between two numeric variables
# to identify the two features that are similar we should take in consideration these factors:
# -- linear relationship (if the points are close to a straight line, the variables have a linear relationship, if the line is steep they have a strong relationship) --
# -- correlation (if the points are clustered along a line, the variables have a strong correlation that might be positive or negative) --
# -- cluster patterns (patterns or clusters can indicate a different grouping in the data  which might suggest similarities between the features) --
# -- no clear patterns (if the points are scattered with no clear pattern, the variables have no relationship) --

# after plotting our data we observe that there are two features that are similar :
# Defense Against the Dark Arts and Astronomy, they form a downward sloping line with a strong negative correlation, which means that as one variable increases the other decreases


def main():
    # Load the data
    data = load_data("data/datasets/dataset_train.csv")
    
    # creating a scatter plot that will answer the question:
    # "what are the two features that are similar ?"
    scatter_plot(data)

if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)