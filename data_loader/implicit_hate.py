import os
import pandas as pd
from datasets import Dataset

class ImplicitHateDataset:
    def __init__(self, local_dir="data"):
        path_to_tsv = os.path.join(local_dir, "implicit_mergedMinority_class_and_implicit_class_lookup_post1.tsv")
        self.data = pd.read_csv(path_to_tsv, sep="\t").dropna(subset=["post", "mergedMinority", "implicit_class"])
    
    def to_hf_dataset(self):
        return Dataset.from_pandas(
            self.data[["post", "mergedMinority", "implicit_class"]].rename(columns={
                "post": "text",
                "mergedMinority": "protected_label",
                "implicit_class": "class_label"
            })
        )
