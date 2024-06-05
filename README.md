# Harry Potter Sorting hat

This project is part of the 42 network cursus (dslr). In this project, we will explore Machine Learning by discovering different tools and implementing a linear classification model, specifically a logistic regression. This project will guide you through the basics of data exploration, visualization, and preprocessing before feeding the data into the machine learning algorithm. The goal is to solve a classification problem using logistic regression.

## Project structure

```
.
|____data_analysis
|____data_preprocessing
| |____data_preprocessing.py
|____environment.yml
|____utils
| |____utils.py
|____scripts
| |____run_data_preprocessing.py
| |____run_data_analysis.py
| |____run_logistic_regression_prediction.py
| |____run_logistic_regression_training.py
| |____run_logistic_regression_evaluation.py
| |____run_data_visualization.py
|____README.md
|____data_visualization
| |____data_visualization.py
|____logistic_regression
| |____logistic_regression.py
|____data
| |____processed
| |____raw
| | |____datasets
| | | |____dataset_train.csv
| | | |____dataset_test.csv
| | |____datasets.tgz

```

### Descrition of Folders and Files

- **`environment.yml`**: Defines the Conda environment with all dependencies.
- **`data/`**: Contains datasets.
  - **`raw/`**: Stores raw, unprocessed data files.
    - **`datasets/`**: Contains training and testing datasets.
    - **`datasets.tgz`**: Compressed raw data files.
  - **`processed/`**: Stores cleaned and processed data files.
- **`data_analysis/`**: Contains scripts for data analysis.
  - **`data_analysis.py`**: Functions for exploratory data analysis.
- **`data_preprocessing/`**: Contains scripts for data cleaning and preprocessing.
  - **`data_preprocessing.py`**: Functions for data preprocessing.
- **`data_visualization/`**: Contains scripts for data visualization.
  - **`data_visualization.py`**: Functions for creating visualizations.
- **`logistic_regression/`**: Contains scripts for logistic regression model.
  - **`logistic_regression.py`**: Implementation of logistic regression, including training, evaluation, and prediction.
- **`scripts/`**: Contains runnable scripts for different stages of the project.
  - **`run_data_preprocessing.py`**: Script to execute data preprocessing.
  - **`run_data_analysis.py`**: Script to execute data analysis.
  - **`run_data_visualization.py`**: Script to generate visualizations.
  - **`run_logistic_regression_training.py`**: Script to train the logistic regression model.
  - **`run_logistic_regression_evaluation.py`**: Script to evaluate the logistic regression model.
  - **`run_logistic_regression_prediction.py`**: Script to make predictions using the logistic regression model.
- **`utils/`**: Contains utility scripts shared across modules.
  - **`utils.py`**: Utility functions.

## Aim of the Project

- Learn how to read a dataset and visualize it in different ways.
- Select and clean unnecessary information from your data.
- Train a logistic regression model to solve a classification problem.
- Create a machine learning toolkit for data exploration and model training.

By following this project structure, you will be able to efficiently navigate through the various stages of data preprocessing, analysis, visualization, and model implementation.
