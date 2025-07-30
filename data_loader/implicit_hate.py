import os
import pandas as pd
from .dataset import Dataset
from datasets import Dataset as HFDataset

class ImplicitHateDataset(Dataset):
    def __init__(self, data_path, split="all"):
        super().__init__(data_path, split)
        self.dataset_name = "implicit_hate"
        self.text_column = "post"
        self.label_column = "implicit_class"
        self.group_column = "mergedMinority"
        self.df = self.load_dataset()

    def load_dataset(self):
        file_path = os.path.join(self.data_path, "implicit_mergedMinority_class_and_implicit_class_lookup_post1.tsv")
        df = pd.read_csv(file_path, sep="\t")
        required_columns = [self.text_column, self.label_column, self.group_column]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset!")
        df = df.dropna(subset=required_columns)
        return df

    def to_hf_dataset(self):
        return HFDataset.from_pandas(
            self.df[[self.text_column, self.label_column, self.group_column]].rename(columns={
                self.text_column: "text",
                self.label_column: "class_label",
                self.group_column: "protected_label"
            })
        )
