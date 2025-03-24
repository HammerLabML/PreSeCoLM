import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from .dataset import CustomDataset, label2onehot
import pickle


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


class BiosDataset(CustomDataset):

    def __init__(self, local_dir: str = None, option: str = 'supervised'):
        super().__init__(local_dir)

        valid_options = ['supervised', 'unsupervised']
        err_str = (("There are two options for the BIOS dataset: 1) for 'unsupervised' the dataset will be loaded from "
                    "huggingface and used as it is, 2) for 'supervised' the dataset will be filtered for valid samples "
                    "and a subset of 10 occupations. Please select one of these options: ")
                   + str(valid_options))
        assert option in valid_options, err_str
        self.option = option

        if option == 'supervised':
            self.name = 'bios_sup'
        else:
            self.name = 'bios'
        self.group_names = ['male', 'female']

        print("load BIOS with option %s" % self.option)
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None):

        if self.option == 'supervised':
            with open(local_dir, 'rb') as handle:
                merged_dataset = pickle.load(handle)

            keys_to_copy = ['hard_text', 'profession', 'gender', 'raw', 'titles_supervised', 'review', 'valid', 'name']
            self.class_names = ['architect', 'surgeon', 'dentist', 'teacher', 'psychologist', 'nurse', 'photographer',
                                'physician', 'attorney', 'journalist']

            # multi-label only reviewed+valid
            ds = filter_bios_dataset(merged_dataset, self.class_names, keys_to_copy, False, True,
                                     True)

            for split in ['train', 'test', 'dev']:
                self.data[split] = ds[split]['hard_text']
                self.labels[split] = np.array(ds[split]['label'])
                self.protected_groups[split] = label2onehot(np.array(ds[split]['gender']))

        else:  # unsupervised
            ds = load_dataset("LabHC/bias_in_bios")

            # class names are not in the meta-data, copied manually from:
            # https://huggingface.co/datasets/LabHC/bias_in_bios
            self.class_names = ['accountant', 'architect', 'attorney', 'chiropractor', 'comedian', 'composer',
                                'dentist', 'dietitian', 'dj', 'filmmaker', 'interior_designer', 'journalist', 'model',
                                'nurse', 'painter', 'paralegal', 'pastor', 'personal_trainer', 'photographer',
                                'physician', 'poet', 'professor', 'psychologist', 'rapper', 'software_engineer',
                                'surgeon', 'teacher', 'yoga_teacher']

            for split in ['train', 'test', 'dev']:
                self.data[split] = ds[split]['hard_text']
                self.labels[split] = np.array(ds[split]['profession']).reshape(-1, 1)
                self.protected_groups[split] = label2onehot(np.array(ds[split]['gender']))

