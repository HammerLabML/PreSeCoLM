import datasets
import numpy as np
import pandas as pd
from .dataset import CustomDataset

class WinoQueer(CustomDataset):

    def __init__(self, local_dir: str = None):
        super().__init__(local_dir)

        self.name = 'winoqueer'
        self.group_names = []

        print('load winoqueer')
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None):

        df = pd.read_csv(local_dir)

        groups_x = set(df['Gender_ID_x'])
        groups_y = set(df['Gender_ID_y'])
        self.group_names = list(groups_x) + list(groups_y)
        n_groups = len(self.group_names)

        n_sent = len(df) * 2
        self.class_names = ['queer', 'normative']

        self.data['test'] = []
        self.labels['test'] = []
        self.protected_groups['test'] = np.zeros((n_sent, n_groups), dtype=float)

        for i in range(len(df)):
            # queer version
            self.data['test'].append(df.loc[i, 'sent_x'])
            self.labels['test'].append(0)
            group = df.loc[i, 'Gender_ID_x']
            self.protected_groups['test'][i*2, self.group_names.index(group)] = 1

            # normative version
            self.data['test'].append(df.loc[i, 'sent_y'])
            self.labels['test'].append(1)
            group = df.loc[i, 'Gender_ID_y']
            self.protected_groups['test'][i*2+1, self.group_names.index(group)] = 1

            
        self.labels['test'] = np.asarray(self.labels['test']).reshape(-1, 1).astype(float)
