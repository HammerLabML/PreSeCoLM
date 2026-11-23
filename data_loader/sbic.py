from .dataset import CustomDataset
import pandas as pd
import datasets
import numpy as np
import itertools

import pandas as pd
import numpy as np

import datasets
import re

from .sbic_helper import load_data_original, load_data_aae_annotation_study, load_data_annotation_study, merge_split, binarize_identity_labels


class SBICDataset(CustomDataset):

    # TODO outside this script
    # copy subset embeddings from full dataset (at least for openai; might need to keep HITIds to make the match or use posts from subset-orig version)
    def __init__(self, local_dir = None, option: str = 'full'):
        super().__init__(local_dir)

        self.name = 'sbic'

        valid_options = ['full', 'subset-orig', 'subset-balanced', 'subset-black']
        err_str = (("There are four options for the SBIC dataset: 1) 'full' the full dataset will be loaded from "
                    "huggingface, alternatively a re-annotated subset can be loaded with different annotation sets:"
                    "2) subset-orig uses original labels (predominantly white annotators, many groups labeled), 3) subset-balanced new labels from an ethnicity-balanced"
                    "annotator pool (again many groups labeled), and 4) subset-black new labels exclusively from black annotaotrs (only covering black and 'other groups'). "
                    "Please select one of these options: ")
                   + str(valid_options))
        assert option in valid_options, err_str

        # these are the merged groups and categories with sufficient support in the subset
        # i.e. more than 10 samples (labeled by at least 1 annotator) in the subset with original labels
        self.group_names = ['white', 'black', 'asian', 'people of color', 'hispanic/ latin-american', 'middle eastern', 'indigenous',
                    'male', 'female', 'TIN', 'lgbtq', 'homosexual',
                    'christian', 'jewish', 'muslim',
                    'mexican', 'american', 'nationality',
                    'physical disability', 'mental health conditions'
                    ]
        # could also use these categories as concepts, but need to load on 'category' level instead of 'merged' then
        # self.category_names = ['race/ethnicity', 'religion', 'nationality', 'gender/ sexual orientation', 'victims', 'disability', 'health']

        # full: keep all samples
        # subset-*: subset from annotation study (TODO link)
        # subset-orig: original labels
        # subset-balanced: study 1 labels
        # subset-black: study 2 labels
        self.option = option 

        self.ids = {} # remember ids for cross-references between subset and full dataset

        if self.option == 'full' or self.option == 'subset-orig':
            self.class_names = ['offensiveYN', 'intentYN', 'whoTarget'] # 3rd means if a group is targeted
        else:
            self.class_names = ['offense', 'intent', 'group_off'] # 3rd means if a group is targeted

        print("load SBIC with option %s and local file: %s" % (option, local_dir))
        self.load(local_dir)
        self.prepare()

    def _set_split(self, df: pd.DataFrame, split: str, text_col: str):
        self.data[split] = df.loc[:, text_col].to_list()
        
        mean_lbl = df.loc[:, self.class_names].to_numpy()
        self.labels[split] = (mean_lbl > 0.5).astype('float64') # could do other label processing / handle as regression

        group_mean_lbl = df.loc[:, self.group_names].to_numpy()
        self.protected_groups[split] = (group_mean_lbl > 0.5).astype('float64') # could use other threshold

        self.ids[split] = list(df.loc[:, 'HITId'])


    def load(self, local_dir=None):
        assert isinstance(local_dir, dict), "SBIC requires a dictionary of local paths"

        # these local files are required for all options
        assert 'label_hierarchy' in local_dir.keys() and 'label_category' in local_dir.keys()
        label_hierarchy_path = local_dir['label_hierarchy']
        label_category_path = local_dir['label_category']

        if 'subset' in self.option:
            key = 'data'
            assert key in local_dir.keys(), "for subset option require at least the path to the data using key: "+key
            data_path = local_dir[key]
        if self.option == 'subset-orig' or self.option == 'full':
            key = 'label_map_orig'
            assert key in local_dir.keys(), "if loading original labels, path to label mapping for original labels is required using key: "+key
            label_lookup_orig_path = local_dir[key]   

        if 'subset' in self.option:
            # start: subset loading

            if self.option == 'subset-orig':
                # get ids from annotation study data
                df = load_data_annotation_study(data_path=data_path, label_hierarchy_path=label_hierarchy_path, label_category_path=label_category_path, 
                                                sel_labels=self.group_names, sel_level='merged', do_aggregate=True, exclude_flagged_samples=True)
                post_ids = list(set(df['HITId']))

                # load original labels for these samples
                df = load_data_original(split='test', label_lookup_path=label_lookup_orig_path, label_hierarchy_path=label_hierarchy_path, 
                                        label_category_path=label_category_path, sel_labels=self.group_names, sel_level='merged', do_aggregate=True,
                                        sel_post_ids=post_ids)
                df = df.sort_values(by='HITId')
                
            elif self.option == 'subset-balanced':
                df = load_data_annotation_study(data_path=data_path, label_hierarchy_path=label_hierarchy_path, label_category_path=label_category_path, 
                                    sel_labels=self.group_names, sel_level='merged', do_aggregate=True, exclude_flagged_samples=True)
                df = df.sort_values(by='HITId')

                # convert <group>_ref to <group> column
                df = df.rename(columns={grp+'_ref': grp for grp in self.group_names})

            else: # subset-black
                df = load_data_aae_annotation_study(data_path=data_path, do_aggregate=True, exclude_flagged_samples=True)
                self.group_names = ['black', 'aae']
                df = df.sort_values(by='HITId')

                # convert B_<lbl> and B_<group>_ref columns
                df = df.rename(columns={'B_black_ref': 'black', 'B_aae': 'aae', 'B_offense': 'offense', 'B_intent': 'intent', 'B_group_off': 'group_off'})

            text_col = 'post' if self.option == 'subset-orig' else 'post_prep'
            self._set_split(df=df, split='test', text_col=text_col)

            
            # end: subset loading
        else:
            for split in ['train', 'test', 'validation']:
                ds = datasets.load_dataset("allenai/social_bias_frames", split=split, trust_remote_code=True)
                df = merge_split(ds, local_dir=local_dir['label_map_orig'])

                # rename target columns
                df = df.rename(columns={'mean_offensiveYN': 'offensiveYN', 'mean_intentYN': 'intentYN', 'mean_whoTarget': 'whoTarget'})

                # binarize group labels (preprocess label format!)
                df['standardized_lbl'] = (df['standardized_lbl'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray)) else x)
                                .str.replace(r"[\[\]'\"]", '', regex=True))
                df, sel_labels = binarize_identity_labels(df, 'standardized_lbl', sel_labels=self.group_names)
                assert sel_labels == self.group_names
                
                if split == 'validation':
                    split = 'dev'
                #df = load_data_original(split=split, label_lookup_path=label_lookup_orig_path, label_hierarchy_path=label_hierarchy_path,
                #                   label_category_path=label_category_path, sel_labels=self.group_names, sel_level='merged', do_aggregate=True)

                text_col = 'post'
                self._set_split(df=df, split=split, text_col=text_col)


