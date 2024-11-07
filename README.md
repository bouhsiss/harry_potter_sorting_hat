# Harry Potter Sorting hat

This project is part of the 42 network cursus (dslr). In this project, we will explore Machine Learning by discovering different tools and implementing a linear classification model, specifically a logistic regression. This project will guide you through the basics of data exploration, visualization, and preprocessing before feeding the data into the machine learning algorithm. The goal is to solve a classification problem using logistic regression.

## Project structure

```
.
|____data_analysis
| |____data_analysis.py
| |____describe.py
|____data_preprocessing
| |____data_preprocessing.py
|____utils
| |____utils.py
|____README.md
|____data_visualization
| |____histogram.py
| |____pair_plot.py
| |____scatter_plot.py
|____logistic_regression
| |____logreg_train.py
| |____logreg_evaluate.py
| |____logreg_predict.py
| |____logistic_regression_weights.txt
|____data
| |____datasets
| | |____dataset_train.csv
| | |____dataset_test.csv
```



## Aim of the Project

- Learn how to read a dataset and visualize it in different ways.
- Select and clean unnecessary information from your data.
- Train a logistic regression model to solve a classification problem.
- Create a machine learning toolkit for data exploration and model training.

## Data Visualization and Interpratation

### Homogenity of Course Scores Across Hogwarts Houses
One of the key steps in our data analysis was to determine the most homogenous distribution of scores across all four houses. To achieve this, we created histograms for each course, segmented by house. Below is a screenshot of the histograms generated:
![Course Scores by Hogwarts House](assets/courses_Scores_by_hogwarts_house.png)

#### Interpretation 

To identify the most homogenous distribution score, we considered the following factors across all four houses:
- **Shape Consistency**: The shape of the distribution is consistent across all houses.
- **Range and Spread**: The range and spread of scores are similar across houses.
- **Central Tendency**: The mean and median are similar across houses.
- **Overlapping Areas**: More overlap indicates more similarity in distributions.

After analyzing the plots, we observed that two courses, **Arithmancy** and **Care of Magical Creatures**, exhibit significant homogeneity. However, **Care of Magical Creatures** shows more homogeneity compared to **Arithmancy**. This is indicated by its narrow range, aligned peaks, and a high degree of overlap, suggesting that the score distributions are more similar across houses. In contrast, **Arithmancy** has a wider range and spread, along with some outliers, indicating less homogeneity.

### Similarity Between Features

Another important analysis involved identifying which features (courses) are similar to each other. This can help in understanding the relationships between different subjects and how they might influence each other.

Below is a screenshot of the scatter plots generated for each pair of courses:

![Scatter Plots of Course Pairs](assets/scatter_plot_of_course_pairs_grouped_by_hogwarts_house.png)

#### Interpretation

Scatter plots are primarily used to observe and show the relationship between two numeric variables. To identify the two features that are similar, we considered the following factors:
- **Linear Relationship**: If the points are close to a straight line, the variables have a linear relationship. If the line is steep, they have a strong relationship.
- **Correlation**: If the points are clustered along a line, the variables have a strong correlation that might be positive or negative.
- **Cluster Patterns**: Patterns or clusters can indicate different groupings in the data, which might suggest similarities between the features.
- **No Clear Patterns**: If the points are scattered with no clear pattern, the variables have no relationship.

After plotting our data, we observed that **Defense Against the Dark Arts** and **Astronomy** are the most similar features. They form a downward sloping line with a strong negative correlation, which means that as one variable increases, the other decreases. This suggests that students who perform well in one of these courses tend to perform less well in the other.


### Selection of features for the Logistic Regression Model


To determine which features to use for the logistic regression model, we created a pair plot to visualize the relationships between the different courses.

Below is a screenshot of the pair plot generated for each pair of courses:

![Pair Plot of Courses](assets/course_pair_plot.png)

#### Interpretation

Pair plots are useful for identifying relationships and correlations between different features. To decide which features to include in our logistic regression model, we considered the following:

- **Redundant Features**: Features that show a strong correlation (positive or negative) are likely redundant. For example, **Defense Against the Dark Arts** and **Astronomy** have a strong negative correlation, so one of them can be dropped.
- **Discriminative Features**: Features that show clear separation between the houses are valuable for classification. Examples include:
  - **Astronomy vs Defense Against the Dark Arts**
  - **Astronomy vs Ancient Runes**
  - **Astronomy vs Charms**
  - **Herbology vs Defense Against the Dark Arts**
  - **Herbology vs Ancient Runes**
  - **Defense Against the Dark Arts vs Ancient Runes**
- **Similar Clusters**: Features that have similar cluster patterns with others might suggest redundancy. For example, **History of Magic** and **Transfiguration** have similar cluster patterns with other features.

Based on these observations, we will focus on the discriminative features and drop redundant ones to improve the performance and efficiency of our logistic regression model. This careful selection helps in creating a more accurate and interpretable model.

## Logistic Regression
In the `logistic_regression` folder, we have implemented the core functionalities for training, evaluating, and predicting using the logistic regression model:

- **Training**: The `logreg_train.py` script is responsible for training the logistic regression model on the processed dataset. We utilized an One-vs-Rest (OvR) logistic regression approach, which allows us to handle multi-class classification problems effectively. The model is trained using mini-batch gradient descent, which helps in optimizing the weights efficiently by processing small batches of data at a time. This method not only speeds up the training process but also helps in achieving better convergence.

- **Weight Application**: After training, we apply the learned weights to the input features. For each class, we compute the predicted probabilities, and to determine the final prediction, we use the `argmax` function. This function selects the class with the highest predicted probability, ensuring that we choose the best prediction based on the model's output.

- **Evaluation**: The `logreg_evaluate.py` script evaluates the performance of the trained model using accuracy. This helps in understanding how well the model performs on unseen data.

- **Prediction**: The `logreg_predict.py` script is used to make predictions on new data using the trained model. It loads the model weights and applies them to the input features to generate predictions.
