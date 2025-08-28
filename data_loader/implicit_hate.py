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


class ImplicitHateDataset(CustomDataset):

    def __init__(self, local_dir: str = None):
        super().__init__(local_dir)

        self.name = 'implicit_hate'
        
        # TODO: specify all groups to be labeled in the dataset (this can be any subset of merged groups for now)
        self.group_names = ['white', 'black', 'asian']

        # TODO: adapt class names for implicitHate
        self.class_names = ['offensiveYN', 'intentYN', 'sexYN']


        print("Loading Implicit Hate dataset from:", local_dir)
        self.load(local_dir)
        self.prepare()

    def _set_split(self, df, split):
        self.data[split] = df['post'].to_list() # TODO column name for text

        # TODO: take the correct column names for the labels
        labels = df[self.class_names].to_numpy()
        self.labels[split] = (labels > 0.5).astype('float64')

	# TODO: check column name
        self.protected_groups[split] = np.zeros((len(df), len(self.group_names)))
        for i, group in enumerate(self.group_names):
            self.protected_groups[split][:, i] = df['mergedMinority'].apply(
                lambda x: 1 if group in str(x).lower() else 0
            ).astype('float64')

    def load(self, local_dir=None):
        for split in ['train', 'test', 'validation']: # whatever the splits are called in the filenames
            # TODO: read the csv files for the current split from disk (the original files that we downloaded)
            # in the 'merge_splits' function the files should be merged (text + labels + mergedMinority into one dataframe)
            # if necessary uncomment these preprocessing lines
            # Ensure label columns are numeric
            for col in self.class_names:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=self.class_names + ['post', 'mergedMinority'])
	    
	    # TODO: similar to the following lines call merge_split with all the dataframes of the current split
            df = merge_split(ds, local_dir)
            all_groups = []
            for sample in df.loc[:,'mergedMinority']:
                for group in sample:
                    if not group in all_groups:
                        all_groups.append(group)
            print(all_groups) # just to see what groups are labeled in the data -> copy to self.group_names

            # possibly rename split
            if split == 'validation':
                split = 'dev'

            # at this point the splits should be named: 'train', 'test' or 'dev'
            self._set_split(df, split)
