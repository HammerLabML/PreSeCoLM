import numpy as np
from datasets import Dataset, DatasetDict


# filter the dataset: from MeasuringFairnessWithBiasedData (TODO: github link)
def filter_bios_dataset(dataset: dict, classes: list, keys_to_copy: list, single_label=True, review_only=True,
                        valid_only=True):
    splits = dataset.keys()
    split_dict = {}
    filtered_dataset = {split: [] for split in splits}
    for split in splits:
        for elem in dataset[split]:
            if valid_only and elem['valid'] != 1:
                continue
            if review_only and elem['review'] != 1:
                continue
            sel_titles = [title for title in elem['titles_supervised'] if title in classes]
            if single_label and len(sel_titles) > 1:
                continue
            if len(sel_titles) == 0:
                continue

            new_entry = {k: elem[k] for k in keys_to_copy}
            if single_label:
                label = classes.index(sel_titles[0])
            else:  # multi-label / one-hot encoded
                label = np.zeros(len(classes))
                for title in sel_titles:
                    label[classes.index(title)] = 1
            new_entry.update({'label': label})
            filtered_dataset[split].append(new_entry)

        cur_split = {k: [elem[k] for elem in filtered_dataset[split]] for k in filtered_dataset[split][0].keys()}
        split_dict[split] = Dataset.from_dict(cur_split, split=split)
    return DatasetDict(split_dict)
