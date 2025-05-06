import os
import sys
import json
import pandas as pd
import yaml

import functools
import getopt
import numpy as np
import scipy

# local imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader
import plotting
import utils

NEUTRAL_KEY = 'neutral'


def normalize(vectors: np.ndarray):
    assert vectors.ndim == 2, "expected a 2d array"
    # normalize each vector of a 2d numpy array
    norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
    vectors = vectors / norms[:, np.newaxis]
    return np.asarray(vectors)


class BiasSpaceModel():
    def __init__(self):
        self.B = None
        self.lbl = []

    @staticmethod
    def group_mean(attribute_set):
        attr_set_norm = normalize(attribute_set)
        center = np.mean(attr_set_norm, axis=0)
        return center

    def compute_bias_space(self, neutral_set: np.ndarray, defining_set_dict: dict, attribute: str = None):
        """
        Computes the bias space based on one attribute. Requires one defining set with neutral
        terms and n >= 1 defining sets representing protected groups. The resulting bias space
        and labels are stacked onto the global bias space/ label list.

        :param neutral_set: numpy array with m neutral term/phrase embeddings of dimension d, shape:(m, d)
        :param defining_set_dict: dict of form {group: defining_set} where defining set is a numpy array of shape (m,d)
        :param attribute: the string identifier of the current attribute, is combined with group labels
        """
        # computes the bias space for one protected attribute/ category given one
        # neutral set of terms and one set of terms per protected group
        n_groups = len(defining_set_dict)
        assert n_groups >= 1, "need at least one protected group!"

        for group, dset in defining_set_dict.items():
            assert len(neutral_set) == len(dset), (
                    "mismatch of defining terms for group %s with neutral terms: %i vs. %i"
                    % (group, len(neutral_set), len(dset)))

        # compute neutral base and 'any' vector
        neutral_base = self.group_mean(neutral_set)

        # in multi-attribute cases, labels can be combined (otherwise multiple 'any' labels would occur!
        lbl_prefix = '%s:' % attribute if attribute is not None else ''

        if n_groups == 1:
            # case1: only one protected group -> only one group vector relative to neutral dir
            group_name = next(iter(defining_set_dict))
            group_dset = next(iter(defining_set_dict.values()))
            print("got only one group (%s): compute group vector relative to neutral base" % group_name)
            bias_comp = self.group_mean(group_dset) - neutral_base
            bias_comp = bias_comp.reshape((1, bias_comp.shape[0]))
            self.B = normalize(bias_comp)
            self.lbl.append(lbl_prefix + group_name)
        else:
            # case2: multiple protected groups -> compute an 'any' vector pointing to the
            #        mean of all groups and a vector for each group
            print("got %i groups: compute 'any' vector and one for each group" % n_groups)

            # any vector is relative to neutral base
            group_mean_vec = np.mean([self.group_mean(dset) for dset in defining_set_dict.values()], axis=0)
            any_dir = group_mean_vec - neutral_base
            bias_comp = [any_dir]

            self.lbl.append(lbl_prefix + 'any')

            # group vectors are relative to the mean of all groups
            for group, dset in defining_set_dict.items():
                other_group_mean_vec = np.mean(
                    [self.group_mean(dset_) for group_, dset_ in defining_set_dict.items() if not group_ == group],
                    axis=0)
                bias_comp.append(self.group_mean(dset) - other_group_mean_vec)
                self.lbl.append(lbl_prefix + group)
            bias_comp = np.asarray(bias_comp)

        # append bias components for current protected attribute to the bias space
        if self.B is None:
            self.B = normalize(bias_comp)
        else:
            print(self.B.shape)
            print(bias_comp.shape)
            self.B = np.vstack([self.B, normalize(bias_comp)])

    def compute_multi_attr_bias_space(self, attribute_dict):
        """
        Computes the bias space from the defining sets for different groups.
        The defining sets of one attribute are equal sized arrays of embeddings.
        Can handle an arbitrary number of protected attributes and requires at least
        one protected group + neutral defining set per attribute.
        The bias spaces are constructed independently for each attribute, then stacked.

        :param attribute_dict: dictionary of the form: {'attributeX': {NEUTRAL_KEY: neutral_def_emb, 'groupY': defining_emb, ...}, ...}
        """

        # reset bias space and labels
        self.B = None
        self.lbl = []

        # test proper structure of attribute dict, then compute bias space per attribute
        assert len(attribute_dict) >= 1, "cannot compute a bias space from an empty dictionary"

        for attr, subdict in attribute_dict.items():
            assert NEUTRAL_KEY in subdict.keys(), "did not provide a neutral def. set for attribute '%s'" % attr
            assert len(subdict) >= 2, "expected defining sets for at least one group plus neutral def. set"

            if len(subdict) == 2:
                self.lbl.append('neutral (%s)' % attr)

            # if multiple attributes are available, combine group and attr label
            attr_str = attr if len(attribute_dict) > 1 else None

            group_def_set_dict = {k: v for k, v in subdict.items() if not k == NEUTRAL_KEY}
            self.compute_bias_space(subdict[NEUTRAL_KEY], group_def_set_dict, attr_str)

        assert self.B is not None, "failed to set bias space"
        assert len(self.B) == len(self.lbl), ("mismatch of bias space dimension and #labels: %i vs. %i"
                                              % (len(self.B), len(self.lbl)))

    def fit(self, attribute_dict: dict):
        """
        Wrapper for compute_multi_attr_bias_space to match the naming of CBM and CAV classes.

        :param attribute_dict: dictionary of the form: {'attributeX': {NEUTRAL_KEY: neutral_def_emb, 'groupY': defining_emb, ...}, ...}
        """
        self.compute_multi_attr_bias_space(attribute_dict)

    def get_concept_vectors(self):
        """
        Returns the bias space (= stacked concept vectors
        :return: bias space = concept vectors
        """
        return self.B

    def get_concept_activations(self, X):
        """
        Returns the concept activation of bias space concepts (= dot product with input)
        :param X: Input embeddings
        :return: Dot product = concept activations
        """
        return np.matmul(X, self.B.T)


def evaluate_bias_space(defining_term_dict, dataset_test, model, pooling, batch_size, emb_dir, plot_dir, file_prefix,
                        local_dir=None, sel_groups_eval=None, sel_groups_bias_space=None):
    sel_groups_eval_ = sel_groups_eval
    if 'any' in sel_groups_eval:
        sel_groups_eval_ = [group for group in sel_groups_eval if not group == 'any']

    # get the data (eval split)
    dataset, emb_defining_attr_dict = utils.get_dataset_with_embeddings(emb_dir, dataset_test, model, pooling,
                                                                        batch_size, local_dir, defining_term_dict)
    # TODO: make sure this works with future datasets
    # TODO: only test split or add dev split?
    eval_split = 'test'
    if len(dataset.splits) == 1:
        eval_split = dataset.splits[0]

    X_test, emb_test, y_test, g_test, _, _ = dataset.get_split(eval_split)
    g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups_eval_, g_test)

    # compute concepts
    bias_space = BiasSpaceModel()
    bias_space.fit(emb_defining_attr_dict)
    # B = bias_space.get_concept_vectors()
    feature_lbl = bias_space.lbl

    if sel_groups_bias_space is None:
        sel_groups_bias_space = feature_lbl

    pred = bias_space.get_concept_activations(emb_test)

    print("feature_lbl: ", feature_lbl)
    print("sel groups bs: ", sel_groups_bias_space)
    print("sel groups eval: ", sel_groups_eval)

    if utils.is1D(g_test):
        print("convert 1d test labels to onehot")
        g_test = data_loader.label2onehot(g_test)
        assert g_test.shape[1] == len(sel_groups_eval), (
                    "shape mismatch %i vs. %i" % (g_test.shape[1], len(sel_groups_eval)))

    add_labels = (sel_groups_eval[0] == 'any')

    # add any- and contrastive labels for further eval
    if add_labels:
        g_true, groups_eval = utils.add_contrastive_any_labels(g_test, sel_groups_eval_)
    else:
        groups_eval = sel_groups_eval
        g_true = g_test

    print("pred: ", pred.shape)
    print("gtest: ", g_true.shape)

    print(sel_groups_eval)
    print(sel_groups_bias_space)
    assert len(sel_groups_eval) == len(sel_groups_bias_space)

    # multiple groups, potentially in different order
    statistics = []
    pvalues = []
    groups_ordered = []
    for test_idx, group in enumerate(groups_eval):
        if add_labels and 'vs. rest' not in group and not group == 'any':
            bias_space_idx = test_idx - len(sel_groups_eval) + 1
        else:
            bias_space_idx = test_idx

        assert type(sel_groups_bias_space[bias_space_idx]) is not list
        feature_match = sel_groups_bias_space[bias_space_idx]
        feature_idx = feature_lbl.index(feature_match)

        corr = scipy.stats.pearsonr(pred[:, feature_idx].flatten(), g_true[:, test_idx].flatten())
        statistics.append(corr.statistic)
        pvalues.append(corr.pvalue)
        groups_ordered.append(feature_match)

    # plot histograms for all test groups (1 or 0) against all predicted concepts, save plot
    model_name = model.replace('/', '_')
    plot_dir = '%s/bias_space_eval' % plot_dir
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if file_prefix is not None:
        plot_savefile = ('%s/%s_%s_%s_%s.png' % (plot_dir, file_prefix, dataset_test, model_name, pooling))
    else:
        plot_savefile = ('%s/%s_%s_%s.png' % (plot_dir, dataset_test, model_name, pooling))
    plotting.plot_feature_histogram(pred, g_true, labels=groups_eval, features=feature_lbl,
                                    xlabel=('Bias Space %s' % file_prefix), ylabel=dataset_test, savefile=plot_savefile)

    return statistics, pvalues, groups_ordered, groups_eval


def run(config):
    # prepare directory where results will be saved
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    # train the CAV models for any combination of LM, pooling and dataset
    for model in models:
        print("at model %s" % model)
        pooling_choices = config["pooling"]
        batch_size = 1
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        if model in openai_models:
            pooling_choices = ['']
        for pool in pooling_choices:
            for dataset in config['datasets']:
                print("dataset: %s" % dataset['name'])
                dataset = utils.get_dataset_with_embeddings(config["embedding_dir"], dataset['name'], model, pool,
                                                            batch_size, local_dir=dataset['local_dir'])


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('compute_embeddings.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('compute_embeddings.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
