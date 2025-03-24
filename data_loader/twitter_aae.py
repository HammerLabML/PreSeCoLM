from .dataset import CustomDataset, label2onehot
from datasets import load_dataset
import numpy as np


class TwitterAAE(CustomDataset):
    # Blodgett et al. (ACL 2018)
    # using the huggingface version: https://huggingface.co/datasets/lighteval/TwitterAAE

    def __init__(self, local_dir: str = None, option: str = 'both'):
        super().__init__(local_dir)
        self.name = 'twitterAAE'
        self.group_names = ['aa', 'white']
        self.unlabeled = True

        print("load twitterAAE")
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None):
        ds_aa = load_dataset('lighteval/TwitterAAE', 'aa')
        ds_white = load_dataset('lighteval/TwitterAAE', 'white')

        # test set only, no labels
        n_per_group = 50000
        self.data['test'] = ds_aa['test']['tweet'] + ds_white['test']['tweet']
        self.labels['test'] = - np.ones((2*n_per_group, 1))
        self.protected_groups['test'] = label2onehot([0 for i in range(n_per_group)] + [1 for i in range(n_per_group)])

