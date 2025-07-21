import os
import pandas as pd
from .dataset import Dataset

# Define a class for handling the Implicit Hate dataset
class ImplicitHateDataset(Dataset):
    def __init__(self, data_path, split="all"):
        super().__init__(data_path, split)
        self.dataset_name = "implicit_hate"  # Name of the dataset
        self.text_column = "post"            # Column containing the text
        self.label_column = "implicit_class" # Column containing the label (class)
        self.group_column = "mergedMinority" # Column indicating the target group
        self.df = self.load_dataset()        # Load and store the dataset as a DataFrame

    # Load the dataset from the specified file
    def load_dataset(self):
        # Construct the full path to the dataset file
        file_path = os.path.join(self.data_path, "implicit_mergedMinority_class_and_implicit_class_lookup_post1.tsv")
        
        # Read the dataset into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\t")

        # Ensure that the required columns exist in the dataset
        required_columns = [self.text_column, self.label_column, self.group_column]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset!")

        # Remove rows with missing values in any of the required columns
        df = df.dropna(subset=required_columns)
        return df
