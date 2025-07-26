from datasets import Dataset
import pandas as pd

class ImplicitHateDataset:
    def __init__(self, path_to_tsv="data/implicit_mergedMinority_class_and_implicit_class_lookup_post1.tsv"):
        # Read the TSV file and drop rows with null values in specific columns
        self.data = pd.read_csv(path_to_tsv, sep="\t").dropna(subset=["post", "mergedMinority", "implicit_class"])
    
    def to_hf_dataset(self):
        # Keep only the required columns and convert the DataFrame to a HuggingFace Dataset
        return Dataset.from_pandas(
            self.data[["post", "mergedMinority", "implicit_class"]].rename(columns={
                "post": "text",
                "mergedMinority": "protected_label",
                "implicit_class": "class_label"
            })
        )

