import pandas as pd
import numpy as np

import datasets
import re


def map_labels(df, lookup_df, label_col='label', new_label_col='standardized_label', 
               lookup_col_raw='raw', lookup_col_target='target', 
               separator=', ', keep_unmapped=False, lower_case_matching=False):
    """
    Maps labels to categories or standardized labels using a lookup table. Assumes label columns with strings of comma separated labels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with the label column
    lookup_df : pandas.DataFrame
        Lookup table with 'raw' and 'standardized' columns
    label_col : str
        Name of the column containing comma-separated labels
    new_label_col : str
        Name of the new column to create with mapped labels
    separator : str
        Separator used in the label strings (default: ', ')
    keep_unmapped : bool
        If True, keep raw labels not found in lookup; otherwise drop them
    lower_case_matching: bool
        If label string should be converted to lower case before matching labels.
    
    Returns:
    --------
    pandas.DataFrame
        Copy of df with additional column including the mapped labels.
    """
    
    # Create lookup dictionary
    lookup_df.loc[lookup_df[lookup_col_target].isna(), lookup_col_target] = None
    label_map = dict(zip(lookup_df[lookup_col_raw], lookup_df[lookup_col_target]))
    
    def map_label_entry(label_string):
        if pd.isna(label_string) or not label_string.strip():
            return ''
        
        # Split, strip whitespace from each label
        if lower_case_matching:
            label_string = label_string.lower()
        raw_labels = [l.strip() for l in label_string.split(separator)]
        
        # Map each label through lookup
        standardized_labels = []
        for raw_label in raw_labels:
            mapped = label_map.get(raw_label)
            if mapped is not None:
                standardized_labels.append(mapped)
            elif keep_unmapped:
                print("error mapping missing for label: ", raw_label)
                print("whole label entry: ", label_string)
                standardized_labels.append(raw_label)  # Keep original
        
        # remove duplicates        
        unique_labels = list(set(standardized_labels))

        try:
            return separator.join(unique_labels)
        except TypeError:
            print("sth went wrong")
            print(unique_labels)
            print(label_string)
    
    # Apply to the dataframe
    result_df = df.copy()
    result_df[new_label_col] = result_df[label_col].apply(map_label_entry)
    
    return result_df

def binarize_identity_labels(df: pd.DataFrame, label_column: str, sel_labels: list[str] = None, target_column_suffix: str = None):
    """
    Converts column of comma separated text labels binary labels (one column per class).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with the label column
    label_col : str
        Name of the column containing comma-separated labels
    target_column_suffix : str
        Optional suffix for target label columns
    
    Returns:
    --------
    pandas.DataFrame
        Copy of df with additional column including the binarized labels.
    """
    df = df.copy()

    if sel_labels is None:
        # use all labels appearing in that column
        sel_labels = (
            df[label_column]
            .str.split(', ')          # split each row into a list
            .explode()                # flatten all lists into one long Series
            .dropna()                 # drop NaNs (from empty strings)
            .str.strip()             # strip whitespace
            .loc[lambda s: s != '']  # drop empty strings
            .unique()
            .tolist()
        )

    new_columns = {}
    for lbl in sel_labels:
        if target_column_suffix is None:
            lbl_col = lbl
        else:
            lbl_col = lbl + target_column_suffix

        pattern = r'\b' + re.escape(lbl) + r'\b'
        new_col = df[label_column].str.contains(
            pattern, 
            na=False, 
            case=False  # Optional: make it case-insensitive
        ).astype(int)
        new_columns[lbl_col] = list(new_col)
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df, sel_labels

def aggregate_annotations(df: pd.DataFrame, target_columns: list[str], sample_id_col: str, keep_columns: list[str]):
    """
    Aggregates annotations over annotators (mean).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with the label column
    target_columns : list[str]
        All columns to be aggregated (target and identity labels - expects numerical values!)
    sample_id_col : str
        Column name of sample ids.
    keep_columns: list[str]
        Other columns to keep in the dataframe (expected to be consistent for all annotators e.g. posts)
    
    Returns:
    --------
    pandas.DataFrame
        Copy of df with additional column including the binarized labels.
    """
    numeric_agg = df.groupby(sample_id_col)[target_columns].mean().reset_index()
    other_content = df[[sample_id_col]+keep_columns].drop_duplicates(subset=[sample_id_col])
    aggregated_df = pd.merge(other_content, numeric_agg, on=sample_id_col, how='left')
    return aggregated_df


def set_target_labels_study1(df: pd.DataFrame):
    """
    Adds simplified target labels based on the raw labels from our first annotation study (base study with all groups), given the following rules:
    - offense if (GOA1 | GOA2 | GOA3 | GOA4 | any offense group)
    - intent if (GOA1 | GOA2 | GOA4)
    - group_off if (GOA1 | GOA2 | any offense group) - GOA3 is excluded because its ambiguous in terms of group offense
    - group_ref if (GRA1 | GRA2 | any reference group)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with the GOA/GRA label columns
    
    Returns:
    --------
    pandas.DataFrame
        Copy of df with additional columns of simplified target labels.
    """
    df = df.copy()

    df['offense'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | (df['GroupOffensive[GOA3]'] == 1) | (df['GroupOffensive[GOA4]'] == 1) | (df['stand_groups_off'] != ''), 'offense'] = 1
    df['intent'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | (df['GroupOffensive[GOA4]'] == 1), 'intent'] = 1
    df['group_off'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | (df['stand_groups_off'] != ''), 'group_off'] = 1
    df['group_ref'] = 0
    df.loc[(df['GroupReferenced[GRA1]'] == 1) | (df['GroupReferenced[GRA2]'] == 1) | (df['stand_groups_ref'] != ''), 'group_ref'] = 1

    return df


def set_target_labels_study2(df: pd.DataFrame):
    """
    Adds simplified target labels based on the raw labels from our second annotation study (asking for black or other group), given the following rules:
    - offense if (GOA1 | GOA2 | GOA3 | GOA4 | any offense group)
    - intent if (GOA1 | GOA2 | GOA4)
    - group_off if (GOA1 | GOA2 | any* offense group) - GOA3 is excluded because its ambiguous in terms of group offense
    - group_ref if (GRA1 | GRA2 | any* reference group)

    *For any offense / reference the 'B_black_*' and 'B_other_*' columns are considered, the 'B_unc_*' column (uncertain about group) not.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with the GOA/GRA label columns
    
    Returns:
    --------
    pandas.DataFrame
        Copy of df with additional columns of simplified target labels.
    """
    df = df.copy()

    df['B_offense'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | (df['GroupOffensive[GOA3]'] == 1) | (df['GroupOffensive[GOA4]'] == 1) | df['B_black_off'] | df['B_other_off'], 'B_offense'] = 1
    df['B_intent'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | (df['GroupOffensive[GOA4]'] == 1), 'B_intent'] = 1
    df['B_group_off'] = 0
    df.loc[(df['GroupOffensive[GOA1]'] == 1) | (df['GroupOffensive[GOA2]'] == 1) | df['B_black_off'] | df['B_other_off'], 'B_group_off'] = 1
    df['B_group_ref'] = 0
    df.loc[(df['GroupReferenced[GRA1]'] == 1) | (df['GroupReferenced[GRA2]'] == 1) | df['B_black_ref'] | df['B_other_ref'], 'B_group_ref'] = 1

    return df


def load_data_annotation_study(data_path: str, label_hierarchy_path: str, label_category_path: str, sel_labels: list[str] = None, sel_level: str = 'standardized',
                               do_aggregate: bool = True, aae_annot_path: str = None, exclude_flagged_samples: bool = False, keep_raw_labels: bool = False):
    """
    Loads the dataset from our annotation study. Applies label hierarchy to derive merged identity groups and category labels as well.
    Offers the option to return all annotations or aggregate over annotators and exclude flagged samples, that potentially do not align 
    with the annotation objective.
    
    Parameters:
    -----------
    data_path : str
        Path to the csv file with post-processed data
    label_hierarchy_path : str
        Path to the csv file with label hierarchy mapping ('standardized_label' to 'merged_label')
    label_category_path : str
        Path to the csv file with label category mapping ('merged_label' to 'category')
    sel_labels : list[str]
        List of identity labels to transform to binary labels. If None all labels in the dataset will be transformed. Defaults to None.
    sel_level : str
        Level of label hierarchy that should be considered for binary labels. Options: 'standardized' (default), 'merged', 'category'.
    do_aggregate : bool
        If true the annotations will be aggregated over annotators (mean labels), otherwise all individual annotations will be returned. Defaults to True.
    data_path : str
        Path to the csv file with post-processed data from AAE study (required if do_aggregate==True). Defaults to None.
    exclude_flagged_samples : bool
        If true all samples flagged during the review process will be removed. Flagged samples may not align with the annotation objective. Defaults to False.
    keep_raw_labels : bool
        If true keep the GOA/GRA columns with raw annotator responses target labels.

    Returns:
    --------
    pandas.DataFrame
        Dataset or subset of the dataset according to specified options.
    """ 
    assert sel_level in ['standardized', 'merged', 'category'], "sel_level must be one of the following choices: 'standardized', 'merged', 'category'"

    # load data and label hierarchy mappings
    df = pd.read_csv(data_path, sep=';')
    label_hierarchy = pd.read_csv(label_hierarchy_path, sep=';')
    label_category_lookup = pd.read_csv(label_category_path, sep=';')

    # exclude flagged if required
    if exclude_flagged_samples:
        df = df[df['flag'] == 0].reset_index()

    # handle special labels (REVIEW, NONE, UNCLEAR)
    # overwrite REVIEW labels with those specified manually
    for column in ['REVIEW lbl', 'stand_groups_ref', 'stand_groups_off']:
        df[column] = df[column].fillna('')

    # remove single NONE and UNCLEAR labels, set group_ref/off label to 0
    ref_no_grp_cond = df['stand_groups_ref'].isin(['NONE', 'UNCLEAR'])
    df.loc[ref_no_grp_cond, 'stand_groups_ref'] = ''
    df.loc[ref_no_grp_cond, 'group_ref'] = 0

    off_no_grp_cond = df['stand_groups_off'].isin(['NONE', 'UNCLEAR'])
    df.loc[off_no_grp_cond, 'stand_groups_off'] = ''
    df.loc[off_no_grp_cond, 'group_off'] = 0

    # remove NONE and UNCLEAR labels (part of a label list)
    def replace_none_unc_groups(s):
        remove_strings = ['NONE, ', ', NONE', 'UNCLEAR, ', ', UNCLEAR']
        for rstr in remove_strings:
            s = s.replace(rstr, '')
        return s
    df['stand_groups_ref'] = df['stand_groups_ref'].apply(replace_none_unc_groups)
    df['stand_groups_off'] = df['stand_groups_off'].apply(replace_none_unc_groups)

    # replace REVIEW labels and clean up lists if replaced by empty string
    def clean_list_string(s):
        """Clean up list formatting: remove empty elements, fix commas."""
        if not s or not isinstance(s, str):
            return s
        parts = [part.strip() for part in s.split(',')]
        parts = [part for part in parts if part]
        return ', '.join(parts)

    df.loc[df['stand_groups_ref'].str.contains('REVIEW'), 'stand_groups_ref'] = df[df['stand_groups_ref'].str.contains('REVIEW')].apply(
        lambda row: clean_list_string(row['stand_groups_ref'].replace('REVIEW', row['REVIEW lbl'])), 
        axis=1)
    df.loc[df['stand_groups_off'].str.contains('REVIEW'), 'stand_groups_off'] = df[df['stand_groups_off'].str.contains('REVIEW')].apply(
        lambda row: clean_list_string(row['stand_groups_off'].replace('REVIEW', row['REVIEW lbl'])), 
        axis=1)

    df = set_target_labels_study1(df)

    # handle inconsistent identity labels:
    # if 'group_ref' but only groups labeled under 'stand_groups_off': assume these are all references (intended annotation logic: off is a subset of ref)
    # if 'group_off' but only groups labeled under 'stand_groups_ref' assume the offense label is correct but annotator forgot to label the group:
    #   if exactly one referenced group: assume this is meant as offense as well - set accordingly
    #   if more referenced groups: unclear which are meant as offense - do nothing
    # if either 'group_ref' or 'group_off' but no group labels at all: cannot make assumptions - do nothing
    cond_ref_insert = (df['group_ref'] == 1) & (df['stand_groups_ref'] == '') & (df['stand_groups_off'] != '')
    cond_off_insert = (df['group_off'] == 1) & (df['stand_groups_off'] == '') & (df['stand_groups_ref'] != '') & (~df['stand_groups_ref'].str.contains(','))
    df.loc[cond_off_insert, 'stand_groups_off'] = df.loc[cond_off_insert, 'stand_groups_ref']
    #df.loc[cond_ref_insert, 'stand_groups_ref'] = df.loc[cond_ref_insert, 'stand_groups_off'] # do not do this (low agreement with other annotators - most groups of these are not labeled by ANYONE else)
    df['insert_groups_ref'] = 0
    df['insert_groups_off'] = 0
    df.loc[cond_ref_insert, 'insert_groups_ref'] = 1
    df.loc[cond_off_insert, 'insert_groups_off'] = 1

    # set flag for possibly implicit mentioned groups
    # when annotators label more groups under offense than reference, they often seem to use ref only for explicitly mentioned groups
    # and off for explicitly and implicitly targeted groups
    cond_implicit_candidates = (df['sum_off'] > df['sum_ref'])
    df['implicit_candidate'] = cond_implicit_candidates
    
    # apply label hierarchy to new dataset
    df = map_labels(df, label_hierarchy, label_col='stand_groups_ref', new_label_col='merged_groups_ref', lookup_col_raw='standardized_label', lookup_col_target='merged_label')
    df = map_labels(df, label_hierarchy, label_col='stand_groups_off', new_label_col='merged_groups_off', lookup_col_raw='standardized_label', lookup_col_target='merged_label')
    df = map_labels(df, label_category_lookup, label_col='merged_groups_ref', new_label_col='categories_ref', lookup_col_raw='merged_label', lookup_col_target='category')
    df = map_labels(df, label_category_lookup, label_col='merged_groups_off', new_label_col='categories_off', lookup_col_raw='merged_label', lookup_col_target='category')

    # binarize identity labels if required
    if sel_level == 'standardized':
        df_bin, sel_labels = binarize_identity_labels(df, 'stand_groups_ref', sel_labels=sel_labels, target_column_suffix='_ref')
        df_bin, sel_labels = binarize_identity_labels(df_bin, 'stand_groups_off', sel_labels=sel_labels, target_column_suffix='_off')
    elif sel_level == 'merged':
        df_bin, sel_labels = binarize_identity_labels(df, 'merged_groups_ref', sel_labels=sel_labels, target_column_suffix='_ref')
        df_bin, sel_labels = binarize_identity_labels(df_bin, 'merged_groups_off', sel_labels=sel_labels, target_column_suffix='_off')
    elif sel_level == 'category':
        df_bin, sel_labels = binarize_identity_labels(df, 'categories_ref', sel_labels=sel_labels, target_column_suffix='_ref')
        df_bin, sel_labels = binarize_identity_labels(df_bin, 'categories_off', sel_labels=sel_labels, target_column_suffix='_off')

    # aggregate over annotators if required (add AAE annotations)
    target_columns = ['offense', 'intent', 'group_off', 'group_ref']
    groups_text_columns = [grp+suffix for grp in ['stand_groups', 'merged_groups', 'categories'] for suffix in ['_ref', '_off']]
    if keep_raw_labels:
        target_columns += ['GroupOffensive[GOA1]', 'GroupOffensive[GOA2]', 'GroupOffensive[GOA3]', 'GroupOffensive[GOA4]', 'GroupOffensive[GOA5]', 
                           'GroupReferenced[GRA1]', 'GroupReferenced[GRA2]', 'GroupReferenced[GRA3]', 'flag', 'review', 'review reason']
        groups_text_columns += ['groups_ref', 'groups_off']
    identity_columns = [grp+suffix for grp in sel_labels for suffix in ['_ref', '_off']]
    numerical_label_columns = target_columns+identity_columns+['implicit_candidate', 'flag']
    if do_aggregate:
        # annotations from first study
        df_agg = aggregate_annotations(df_bin, target_columns=numerical_label_columns, sample_id_col='sample_id', keep_columns=['post_prep', 'HITId'])

        if aae_annot_path is not None:
            # annotations from aae/ black study
            print("add aae data")

            df_aae = pd.read_csv(aae_annot_path, sep=';')
            df_aae = set_target_labels_study2(df_aae)
            grp_cols = ['B_black_ref', 'B_other_ref',  'B_black_off', 'B_other_off', 'B_unc_ref', 'B_unc_off']
            for col in grp_cols:
                df_aae[col] = df_aae[col].fillna(0).astype(int)

            aae_num_cols = ['B_offense', 'B_intent', 'B_group_ref', 'B_group_off', 'B_black_ref', 'B_other_ref',  'B_black_off', 'B_other_off', 
                        'B_unc_ref', 'B_unc_off', 'AAE (auto)', 'post_prep_aae', 'post_prep_tokendiff', 'B_aae', 'B_aae_imitate', 'B_aae_unc']
            df_agg_aae = aggregate_annotations(df_aae, target_columns=aae_num_cols, sample_id_col='sample_id', keep_columns=['post_prep', 'HITId'])

            # merge both
            df_agg = df_agg.merge(
                df_agg_aae[['sample_id']+aae_num_cols], 
                on='sample_id', 
                how='left'
            )

        return df_agg
    else:
        df_bin = df_bin.loc[:,['annotator_uuid', 'sample_id', 'HITId', 'review', 'flag', 'flag reason', 'post_prep', 'implicit_candidate', 'insert_groups_off', 'insert_groups_ref', 'InGroup', 'InGroupWhich', 'InGroupWhy']+target_columns+identity_columns+groups_text_columns]
        return df_bin


def load_data_aae_annotation_study(data_path: str, do_aggregate: bool = True, exclude_flagged_samples: bool = False, keep_raw_labels: bool = False):
    """
    Loads the dataset from our annotation study. Applies label hierarchy to derive merged identity groups and category labels as well.
    Offers the option to return all annotations or aggregate over annotators and exclude flagged samples, that potentially do not align 
    with the annotation objective.
    
    Parameters:
    -----------
    data_path : str
        Path to the csv file with post-processed data
    do_aggregate : bool
        If true the annotations will be aggregated over annotators (mean labels), otherwise all individual annotations will be returned. Defaults to True.
    exclude_flagged_samples : bool
        If true all samples flagged during the review process will be removed. Flagged samples may not align with the annotation objective. Defaults to False.
    keep_raw_labels : bool
        If true keep the GOA/GRA columns with raw annotator responses target labels.

    Returns:
    --------
    pandas.DataFrame
        Dataset or subset of the dataset according to specified options.
    """
    df_aae = pd.read_csv(data_path, sep=';')
    df_aae = set_target_labels_study2(df_aae)

    if exclude_flagged_samples:
        df_aae = df_aae[df_aae['flag'] == 0]

    # columns to keep
    keep_cols = ['annotator_uuid', 'sample_id', 'HITId', 'post_prep'] #, 'review', 'flag', 'flag reason', 'post_prep']
    numeric_cols = ['B_offense', 'B_intent', 'B_group_ref', 'B_group_off', 'B_black_ref', 'B_other_ref',  'B_black_off', 'B_other_off', 
                    'B_unc_ref', 'B_unc_off', 'AAE (auto)', 'post_prep_aae', 'post_prep_tokendiff', 'B_aae', 'B_aae_imitate', 'B_aae_unc']

    if do_aggregate:
        keep_cols = ['post_prep', 'HITId']
        df_agg_aae = aggregate_annotations(df_aae, target_columns=numeric_cols, sample_id_col='sample_id', keep_columns=keep_cols)
        return df_agg_aae.loc[:,keep_cols+numeric_cols]

    if keep_raw_labels:
        numeric_cols += ['GroupOffensive[GOA1]', 'GroupOffensive[GOA2]', 'GroupOffensive[GOA3]', 'GroupOffensive[GOA4]', 'GroupOffensive[GOA5]', 
                        'GroupReferenced[GRA1]', 'GroupReferenced[GRA2]', 'GroupReferenced[GRA3]', 'AAE', 'flag', 'review', 'review reason']

    return df_aae.loc[:, keep_cols+numeric_cols]

        


def handle_intersectional_groups(df: pd.DataFrame, label_col: str, target_col: str):
    df = df.copy()
    intersec_group_dict = {
        'trans': ['trans men', 'trans women'],
        'homosexual': ['gay men', 'lesbian women'],
        'bisexual': ['bisexual men', 'bisexual women']
    }

    def map_intersec_group(s):
        for grp, subgroups in intersec_group_dict.items():
            if subgroups[0] in s and subgroups[1] in s:
                combined = subgroups[1]+', '+subgroups[0]
                assert combined in s, s
                s = s.replace(combined, grp)
        return s

    df[target_col] = df[label_col].apply(map_intersec_group)

    return df
    


def load_data_original(split: str, label_lookup_path: str, label_hierarchy_path: str, label_category_path: str, sel_post_ids: list = None, 
                       sel_labels: list[str] = None, sel_level: str = 'standardized', do_aggregate: bool = True):
    """
    Loads a split of the original dataset. Applies label hierarchy to derive merged identity groups and category labels as well.
    Offers the option to return all annotations or aggregate over annotators.
    
    Parameters:
    -----------
    split : str
        Which split (train/ validation/ test) to load.
    label_lookup_path : str
        Path to the csv file label mapping (raw freetext labels to standardized labels)
    label_hierarchy_path : str
        Path to the csv file with label hierarchy mapping ('standardized_label' to 'merged_label')
    label_category_path : str
        Path to the csv file with label category mapping ('merged_label' to 'category')
    sel_post_ids: list
        List of post ids. Defaults to None. If not none use only the subset of posts with the specified ids.
    sel_labels : list[str]
        List of identity labels to transform to binary labels. If None all labels in the dataset will be transformed. Defaults to None.
    sel_level : str
        Level of label hierarchy that should be considered for binary labels. Options: 'standardized' (default), 'merged', 'category'.
    do_aggregate : bool
        If true the annotations will be aggregated over annotators (mean labels), otherwise all individual annotations will be returned. Defaults to True.
    exclude_flagged_samples : bool
        If true all samples flagged during the review process will be removed. Flagged samples may not align with the annotation objective. Defaults to False.
    
    Returns:
    --------
    pandas.DataFrame
        Dataset or subset of the dataset according to specified options.
    """
    assert sel_level in ['standardized', 'merged', 'category'], "sel_level must be one of the following choices: 'standardized', 'merged', 'category'"

    ds = datasets.load_dataset("allenai/social_bias_frames", split=split, trust_remote_code=True)
    df = pd.DataFrame(ds)

    label_lookup = pd.read_csv(label_lookup_path, sep=';')
    label_hierarchy = pd.read_csv(label_hierarchy_path, sep=';')
    label_category_lookup = pd.read_csv(label_category_path, sep=';')

    # convert the annotation columns to numeric
    target_columns = ['offensiveYN', 'intentYN', 'sexYN', 'whoTarget']
    for col in target_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # remove samples with nan in offense rating (not clear why nan / how to impute)
    df = df[~df['offensiveYN'].isna()]

    # nans in whoTarget are imputed with 0 (can happen because this question depended on whether the post is offensive)
    df['whoTarget'] = df['whoTarget'].fillna(0)

    # group label preprocessing (keep freetext and handle intersectional groups)
    df = df.rename(columns={'targetMinority': 'groups'})
    df = handle_intersectional_groups(df, label_col='groups', target_col='groups')

    df = map_labels(df, label_lookup, label_col='groups', new_label_col='stand_groups', lookup_col_raw='raw', lookup_col_target='standardized_lbl', lower_case_matching=True)

    # get only samples with specified post ids
    if sel_post_ids is not None:
        df = df[df['HITId'].isin(sel_post_ids)]

    # remove duplicate entries (where the combination of workerId and HITId is duplicate; individual workerIds and HITIds can occur multiple times of course!)
    df = df.drop_duplicates(subset=['HITId', 'WorkerId'], keep='first').reset_index()

    # apply label hierarchy to old dataset
    df = map_labels(df, label_hierarchy, label_col='stand_groups', new_label_col='merged_groups', lookup_col_raw='standardized_label', lookup_col_target='merged_label')
    df = map_labels(df, label_category_lookup, label_col='merged_groups', new_label_col='categories', lookup_col_raw='merged_label', lookup_col_target='category')
    
    # binarize identity labels if required
    if sel_level == 'standardized':
        df_bin, sel_labels = binarize_identity_labels(df, 'stand_groups', sel_labels=sel_labels)
    elif sel_level == 'merged':
        df_bin, sel_labels = binarize_identity_labels(df, 'merged_groups', sel_labels=sel_labels)
    elif sel_level == 'category':
        df_bin, sel_labels = binarize_identity_labels(df, 'categories', sel_labels=sel_labels)

    # aggregate over annotators if required
    identity_columns = sel_labels
    numerical_label_columns = target_columns+sel_labels
    if do_aggregate:
        df_agg = aggregate_annotations(df_bin, target_columns=numerical_label_columns, sample_id_col='HITId', keep_columns=['post'])
        return df_agg
    else:
        df_bin = df_bin.loc[:,['WorkerId', 'HITId', 'post']+target_columns+sel_labels+['groups', 'stand_groups', 'merged_groups', 'categories']]
        return df_bin


def load_both_datasets(new_data_path: str, label_lookup_orig_path: str, label_hierarchy_path: str, label_category_path: str, sel_labels: list, sel_level: str, 
                       do_aggregate: bool, aae_annot_path: str = None, exclude_flagged_samples: bool = False):
    """
    Loads the new and original dataset. Both will be loaded with the same options as specified in the parameters. 
    Likewise only the subset of the original dataset's test set used in our annotation study is loaded.
    
    Parameters:
    -----------
    new_data_path: str
        Path to new dataset (csv file).
    label_lookup_orig_path : str
        Path to the csv file label mapping (raw freetext labels to standardized labels)
    label_hierarchy_path : str
        Path to the csv file with label hierarchy mapping ('standardized_label' to 'merged_label')
    label_category_path : str
        Path to the csv file with label category mapping ('merged_label' to 'category')
    sel_labels : list[str]
        List of identity labels to transform to binary labels. If None all labels in the dataset will be transformed. Defaults to None.
    sel_level : str
        Level of label hierarchy that should be considered for binary labels. Options: 'standardized' (default), 'merged', 'category'.
    do_aggregate : bool
        If true the annotations will be aggregated over annotators (mean labels), otherwise all individual annotations will be returned. Defaults to True.
    exclude_flagged_samples : bool
        If true all samples flagged during the review process will be removed. Flagged samples may not align with the annotation objective. Defaults to False.
    
    Returns:
    --------
    pandas.DataFrame, pandas.DataFrame
        New and original dataset loaded and processed according to the specified options.
    """
    df_new = load_data_annotation_study(data_path=new_data_path, label_hierarchy_path=label_hierarchy_path, 
                                        label_category_path=label_category_path, sel_labels=sel_labels, 
                                        sel_level=sel_level, do_aggregate=do_aggregate, aae_annot_path=aae_annot_path,
                                        exclude_flagged_samples=exclude_flagged_samples)
    post_ids = list(set(df_new['HITId']))
    df_orig = load_data_original(split='test', label_lookup_path=label_lookup_orig_path, label_hierarchy_path=label_hierarchy_path, 
                                 label_category_path=label_category_path, sel_labels=sel_labels, sel_level=sel_level,
                                 do_aggregate=do_aggregate, sel_post_ids=post_ids)
    return df_new, df_orig


