import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data
import sys

def pair_plot(data):
    # List of courses to plot
    courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]


    # define colors for each house
    house_colors = {
        "Ravenclaw": "#0e1a40",
        "Slytherin": "#1a472a",
        "Gryffindor": "#740001",
        "Hufflepuff": "#ecb939"

    }

    # Create the pair plot and distinguish the houses (target variable) by color
    sns.pairplot(data, vars=courses, hue='Hogwarts House', palette=house_colors, diag_kind='kde')

    plt.show()

# interpretation
# TO BE DROPPED (redundant features) : -- defense against the dark arts and astronomy -- have a strong negative correlation which means they're redundant features so we can drop one of them
# TO BE KEPT (discrimantive features) : -- astronomy vs defense against the dark arts, astronomy vs ancient runes, astronomy vs charms, herblogy vs defense against the dark arts, herobolgy vs ancient runes, defense against the dark arts vs ancient runes -- these features have a clear separation between the houses which makes them discriminative features
# TO BE CONSIDERED (similar clusters) : -- history of magic and transfiguration -- have a similar pattern of clusters with other features which might suggest a redundancy or a similarity in the data


def main():
    # Load the data
    data = load_data("data/datasets/dataset_train.csv")

    # creating a pair plot that will answer the question:
    # "from the visualization, what features are we going to use for the logistic regression model ?"
    pair_plot(data)

if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)