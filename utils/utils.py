import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file write its dimensions and return it as a \
pandas DataFrame. return None if an error occurs.

    Parameters:
        path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        assert isinstance(path, str), "Path must be a string"
        dataset = pd.read_csv(path)
        return dataset
    except (FileNotFoundError, pd.errors.ParserError, AssertionError) as e:
        print("Error loading dataset: {0}".format(e))
        return None

