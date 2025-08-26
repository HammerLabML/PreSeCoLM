from .dataset import CustomDataset
import pandas as pd
import datasets
import numpy as np
import itertools

# TODO: remove unless needed here too
def merge_minority_lists(series):
    all_items = set()
    for sublist in series.dropna():
        items = str(sublist).split(',')
        all_items.update([item.strip() for item in items])
    return ', '.join(sorted(all_items))

# TODO: similar function for merging the ImplicitHate dataset
# TODO: pass all dataframes for the current split (als pandas dataframe)
def merge_split(ds, local_dir):
    df = pd.DataFrame(ds)

    # Convert the annotation columns to numeric
    for col in ['offensiveYN', 'intentYN', 'sexYN']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, NaNs if conversion fails

    # Compute mean values per HITId
    mean_df = df.groupby('HITId')[['offensiveYN', 'intentYN', 'sexYN']].transform('mean')

    # Compute merged targetMinority per HITId
    df['merged_targetMinority'] = df.groupby('HITId')['targetMinority'].transform(merge_minority_lists)

    # Add the mean columns back to the original DataFrame
    df['mean_offensiveYN'] = mean_df['offensiveYN']
    df['mean_intentYN'] = mean_df['intentYN']
    df['mean_sexYN'] = mean_df['sexYN']

    # Load overview data
    overview_df = pd.read_csv(local_dir)

    # Make sure both columns are strings
    overview_df['targetMinority'] = overview_df['targetMinority'].astype(str)
    overview_df['mergedMinority'] = overview_df['mergedMinority'].astype(str)

    # Mapping dictionary: targetMinority -> mergedMinority
    minority_map = dict(zip(overview_df['targetMinority'], overview_df['mergedMinority']))

    # Function to map multiple targetMinorities to mergedMinority values
    def map_merged_minorities(targets):
        if pd.isna(targets):
            return ''
        groups = [grp.strip() for grp in str(targets).split(', ')]
        merged = [minority_map.get(g, '').split(', ') for g in groups]
        merged = list(itertools.chain.from_iterable(merged))
        if 'nan' in merged:
            merged.remove('nan')
        return list(set(merged))

    # Apply mapping function
    df['mergedMinority'] = df['merged_targetMinority'].apply(map_merged_minorities)

    merged = df.groupby('HITId', as_index=False)[
        ['post', 'mean_offensiveYN', 'mean_intentYN', 'mean_sexYN', 'mergedMinority']].first()

    return merged


class SBICDataset(CustomDataset):

    def __init__(self, local_dir: str = None):
        super().__init__(local_dir)

        self.name = 'sbic'
        
        # TODO: specify all groups to be labeled in the dataset (this can be any subset of merged groups for now)
        self.group_names = ['white', 'black', 'asian']

        # TODO: adapt class names for implicitHate
        self.class_names = ['offensiveYN', 'intentYN', 'sexYN']

        print("load ImplicitHate with local file: %s" % local_dir)
        self.load(local_dir)
        self.prepare()

    def _set_split(self, df, split):
        self.data[split] = df.loc[:, 'post'].to_list() # TODO column name for text

        # TODO: adapt column names for implicit hate (if there are multiple annotators, otherwise just remove
        #       and take the regular labels (as numpy array)
        mean_lbl = df.loc[:, ['mean_offensiveYN', 'mean_intentYN', 'mean_sexYN']].to_numpy()
        self.labels[split] = (mean_lbl > 0.5).astype('float64')
        # filter uncertain labels? around ~15% of samples are between 0.4 and 0.6

        self.protected_groups[split] = np.zeros((len(df), len(self.group_names)))
        for i, label in enumerate(self.group_names):
            self.protected_groups[split][:, i] = df['mergedMinority'].apply(lambda x: 1 if label in x else 0).astype('float64') # TODO: check column name

    def load(self, local_dir=None):
        for split in ['train', 'test', 'validation']: # whatever the splits are called in the filenames
            # TODO: instead of loading from datasets lib read all the files from disk for the current split
            ds = datasets.load_dataset("allenai/social_bias_frames", split=split, trust_remote_code=True)

            # TODO: add the function for merging all files and protected groups and call it here
            df = merge_split(ds, local_dir)
            all_groups = []
            for sample in df.loc[:,'mergedMinority']:
                for group in sample:
                    if not group in all_groups:
                        all_groups.append(group)
            print(all_groups) # just to see what groups are labeled in the data -> copy to self.group_names

            # possible rename split
            if split == 'validation':
                split = 'dev'

            # the splits should be named: 'train', 'test', 'dev'
            self._set_split(df, split)


