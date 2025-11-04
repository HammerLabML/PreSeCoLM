from .dataset import CustomDataset
import pandas as pd
import datasets
import numpy as np
import itertools

SEL_GROUPS = ["female", "male", "homosexual", "lgbtq+", 
              "latino/latina", "middle eastern", "asian", "non-white", "white", "black",
              "progressive", "conservative", "liberal",
              "illegal", "immigrant", "minorities", 
              "jewish", "muslim/islam", "religion"
]
NEGATIVE_CLASS = "not_hate"


def load_and_merge(local_dir: str, verbose: bool = False):
    # assuming the structure from the downloaded zip file (tsv files)
    # and the merge group lookup csv file in the same directory
    stage1_file = local_dir + 'implicit_hate_v1_stg1_posts.tsv'
    stage2_file = local_dir + 'implicit_hate_v1_stg2_posts.tsv'
    stage3_file = local_dir + 'implicit_hate_v1_stg3_posts.tsv'
    group_lookup_file = local_dir + 'merged_group_lookup.csv'

    stage1 = pd.read_csv(stage1_file, sep='\t')
    stage2 = pd.read_csv(stage2_file, sep='\t')
    stage3 = pd.read_csv(stage3_file, sep='\t')
    lookup = pd.read_csv(group_lookup_file, sep=',')

    # create dataframe for merge
    merged = stage1.copy()
    merged['implicit_class'] = ''
    merged['extra_implicit_class'] = ''
    merged['target raw'] = ''
    merged['implied_statement'] = ''
    merged['target merged'] = ''

    # merge implicit class
    for i in range(len(stage2)):
        stage2_row = stage2.iloc[i]
        merged.loc[merged['post'] == stage2_row['post'], 'implicit_class'] = stage2_row['implicit_class']
        if pd.notna(stage2_row['extra_implicit_class']):
            merged.loc[merged['post'] == stage2_row['post'], 'extra_implicit_class'] = stage2_row['extra_implicit_class']

    # merge target groups
    all_groups = {}
    for i in range(len(stage3)):
        stage3_row = stage3.iloc[i]
        merged.loc[merged['post'] == stage3_row['post'], 'implied_statement'] = stage3_row['implied_statement']

        if pd.notna(stage3_row['target']):
            target = str(stage3_row['target']).lower()
            merged.loc[merged['post'] == stage3_row['post'], 'target raw'] = target
            lookup_filt = lookup[lookup['target_normalized'] == target]
            if len(lookup_filt) != 1:
                print(len(lookup_filt), target)
            else:
                group_lbl = str(lookup_filt.loc[lookup_filt.index[0],'mergedMinority'])
                if group_lbl == 'nan':
                    continue
                merged.loc[merged['post'] == stage3_row['post'], 'target merged'] = group_lbl
                groups = group_lbl.replace(', ', ',').split(',')
                for grp in groups:
                    if grp in all_groups.keys():
                        all_groups[grp] += 1
                    else:
                        all_groups[grp] = 1
    
    if verbose:
        print("protected groups")
        print(all_groups)

    return merged
    

class ImplicitHate(CustomDataset):

    def __init__(self, local_dir: str = None, option: str = 'all'):
        super().__init__(local_dir)

        self.name = 'implicit_hate'
        self.group_names = SEL_GROUPS
        self.class_names = []
        self.option = option # all: use all samples, positive-only: only positive samples (implicit hate)

        print("Loading Implicit Hate dataset with option %s from: %s" % (option, local_dir))
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None, verbose=False):
        df = load_and_merge(local_dir, verbose)

        if self.option == 'positive-only':
            df = df[df['class'] != NEGATIVE_CLASS]
        
        # using major class (hate / not hate) and implicit classes
        self.class_names = list(set(df['class'])) + list(set(df['implicit_class']))
        self.class_names.remove('')

        self.data['test'] = list(df['post'])
        self.labels['test'] = np.zeros((len(df), len(self.class_names)))
        self.protected_groups['test'] = np.zeros((len(df), len(self.group_names)))

        for i, idx in enumerate(df.index):
            # major label (hate / not hate)
            self.labels['test'][i, self.class_names.index(df.loc[idx, 'class'])] = 1

            # implicit class labels
            if pd.notna(df.loc[idx, 'implicit_class']) and df.loc[idx, 'implicit_class'] != '':
                self.labels['test'][i, self.class_names.index(df.loc[idx, 'implicit_class'])] = 1
            if pd.notna(df.loc[idx, 'extra_implicit_class']) and df.loc[idx, 'extra_implicit_class'] != '':
                self.labels['test'][i, self.class_names.index(df.loc[idx, 'extra_implicit_class'])] = 1

            # protected groups
            groups = df.loc[idx, 'target merged']
            for group in groups.replace(', ', ',').split(','):
                if group in self.group_names:
                    self.protected_groups['test'][i, self.group_names.index(group)] = 1
