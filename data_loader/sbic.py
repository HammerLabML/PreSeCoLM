from .dataset import CustomDataset
import pandas as pd
import datasets
import numpy as np
import itertools


def merge_minority_lists(series):
    all_items = set()
    for sublist in series.dropna():
        items = str(sublist).split(',')
        all_items.update([item.strip() for item in items])
    return ', '.join(sorted(all_items))


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
        # these groups do not appear in all splits:
        # hetero, mormon, cis, hindu, asexual, feminists, pagan/atheist, blind

        # very rare (<0.1% in test data):
        # mixed race, non-binary, autism, activists, police, accident/ natural disaster
        self.group_names = ['white', 'black', 'asian', 'non-white', 'latin-american', 'hispanic', 'mixed race', 'middle eastern', 'indigenous',
                            'male', 'female', 'non-binary', 'trans', 'lgbtq+',
                            'bisexual', 'homosexual',
                            'christian', 'jewish', 'muslim/islam', 'religion',
                            'physical illness/ disorder', 'mental illness/ disorder', 'physical disability', 'mental disability', 'autism',
                            'overweight', 'children', 'minors', 'old people', 'bad looking',
                            'poor', 'political group', 'feminist', 'liberal', 'conservatives', 'activists', 'police',
                            'violence victims', 'sexual assault/harassment victims', 'holocaust victims', 'genocide victims', 'terrorism victims', 'shooting victims', 'accident/ natural disaster victims']

        self.class_names = ['offensiveYN', 'intentYN', 'sexYN']

        print("load SBIC with local file: %s" % local_dir)
        self.load(local_dir)
        self.prepare()

    def _set_split(self, df, split):
        self.data[split] = df.loc[:, 'post'].to_list()

        mean_lbl = df.loc[:, ['mean_offensiveYN', 'mean_intentYN', 'mean_sexYN']].to_numpy()
        self.labels[split] = (mean_lbl > 0.5).astype('int')
        # TODO: filter uncertain labels? around ~15% of samples are between 0.4 and 0.6

        self.protected_groups[split] = np.zeros((len(df), len(self.group_names)), dtype=int)
        for i, label in enumerate(self.group_names):
            self.protected_groups[split][:, i] = df['mergedMinority'].apply(lambda x: 1 if label in x else 0)

    def load(self, local_dir=None):
        for split in ['train', 'test', 'validation']:
            ds = datasets.load_dataset("allenai/social_bias_frames", split=split)
            df = merge_split(ds, local_dir)
            if split == 'validation':
                split = 'dev'
            self._set_split(df, split)


