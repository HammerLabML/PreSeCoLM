import os
import sys
import json
import pickle
from tqdm import tqdm
import pandas as pd
import yaml

import functools
import getopt
import numpy as np
import math
import random
import scipy
from sklearn.metrics import f1_score, precision_score, recall_score

from pie_data import get_dataset, label2onehot
import plotting
import utils

class BiasSpaceModel():

    def __init__(self):
        self.B = None
        self.lbl = None

    def normalize(self, vectors: np.ndarray):
        norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
        vectors = vectors / norms[:, np.newaxis]
        return np.asarray(vectors)
    
    def attr_mean(self, attribute_set):
        A_unit = self.normalize(attribute_set)
        center = np.mean(A_unit, axis=0)
        return center

    def get_bias_space(self, attribute_sets):
        n = len(attribute_sets)
        assert n >= 2, "need at least two attribute groups to measure bias!"

        if n == 2:
            print("got two attriubtes:")
            print(self.attr_mean(attribute_sets[0]).shape)
            bias_components = np.asarray([self.attr_mean(attribute_sets[0]) - self.attr_mean(attribute_sets[1]), self.attr_mean(attribute_sets[1]) - self.attr_mean(attribute_sets[0])])
        else:
            print("got n > 2  attributes")
            print(self.attr_mean(attribute_sets[0]).shape)
            bias_components = np.asarray([self.attr_mean(attribute_sets[i]) - np.mean([self.attr_mean(attribute_sets[j]) for j in range(0,n) if j != i], axis=0) for i in range(0,n)])
        return self.normalize(bias_components)
    
    def stack_bias_spaces(self, bias_spaces : list, labels: dict):
        assert len(bias_spaces)==len(labels)

        bias_space = np.vstack(bias_spaces)
        label_stack = [lbl for k,v in labels.items() for lbl in v]
    
        return bias_space, label_stack
    
    def compute_multi_attr_bias_space(self, attribute_dict, attr_embs):
        subspaces = [self.get_bias_space(attr_emb) for attr_emb in attr_embs]
        return self.stack_bias_spaces(subspaces, attribute_dict)
        
    def fit(self, samples_by_group, attr_dict):
        self.B, self.lbl = self.compute_multi_attr_bias_space(attr_dict, samples_by_group)

    def get_concept_vectors(self):
        return self.B

    def get_concept_activations(self, X):
        protected_features = utils.get_features(X, self.B)
        return protected_features

def evaluate_bias_space(defining_terms, attribute_dict, dataset_test, model, pooling, batch_size, emb_dir, plot_dir, file_prefix, local_dir=None, sel_groups_eval=None, sel_groups_bias_space=None):
    _, _, _, _, X_test, emb_test, y_test, g_test, groups, emb_defining_attr, _ = utils.get_dataset_and_embeddings(emb_dir, dataset_test, model, pooling, batch_size, local_dir, defining_terms=defining_terms, sel_groups=sel_groups_eval)
    bias_space = BiasSpaceModel()
    bias_space.fit(emb_defining_attr, attribute_dict)
    #B = bias_space.get_concept_vectors()
    feature_lbl = bias_space.lbl# [lbl.split(':')[1] for lbl in ccm.lbl

    if sel_groups_bias_space is None:
        sel_groups_bias_space = feature_lbl
    
    pred = bias_space.get_concept_activations(emb_test)

    if utils.is1D(g_test) and len(sel_groups_eval) > 1:
        print("convert 1d test labels to onehot")
        g_test = label2onehot(g_test)
        assert len(sel_groups_eval) == g_test.shape[1]

    print(sel_groups_eval)
    print(sel_groups_bias_space)
    assert len(sel_groups_eval) == len(sel_groups_bias_space)

    if len(sel_groups_eval) == 1:
        feature_match = sel_groups_bias_space[0]
        feature_idx = feature_lbl.index(feature_match)
        corr = scipy.stats.pearsonr(pred[:, feature_idx].flatten(), g_test)
        statistics = [corr.statistic]
        pvalues = [corr.pvalue]
        groups_ordered = sel_groups_bias_space
    else:
        # multiple groups, potentially in different order
        statistics = []
        pvalues = []
        groups_ordered = []
        for test_idx, group in enumerate(sel_groups_eval):
            assert type(sel_groups_bias_space[test_idx]) != list

            feature_match = sel_groups_bias_space[test_idx]
            feature_idx = feature_lbl.index(feature_match)

            corr = scipy.stats.pearsonr(pred[:,feature_idx].flatten(), g_test[:,test_idx].flatten())
            statistics.append(corr.statistic)
            pvalues.append(corr.pvalue)
            groups_ordered.append(feature_match)

    # plot histograms for all test groups (1 or 0) against all predicted concepts, save plot
    model_name = model.replace('/','_')
    plot_dir = '%s/bias_space_eval' % plot_dir
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if file_prefix is not None:
        plot_savefile = ('%s/%s_%s_%s_%s.png' % (plot_dir, file_prefix, dataset_test, model_name, pooling))
    else:
        plot_savefile = ('%s/%s_%s_%s.png' % (plot_dir, dataset_test, model_name, pooling))
    plotting.plot_feature_histogram(pred, g_test, labels=sel_groups_eval, features=feature_lbl, xlabel=('Bias Space %s' % file_prefix), ylabel=dataset_test, savefile=plot_savefile)

    return statistics, pvalues, groups_ordered, sel_groups_eval


def run_bias_space_eval(config):
    # prepare directory where results will be saved
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])

    # config with evaluation setups (data_loader, protected attributes and groups...)
    with open(config['bias_space_eval_config'], 'r') as stream:
        eval_setups_by_attr = yaml.safe_load(stream)

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    result_keys = ["protected_attr", "dataset (train)", "dataset (eval)", "embedder", "pooling", "group (test)",
                       "concept (train)", "F1", "Pearson R", "Pearson p"]

    # read existing results or create new dataframe
    results_path = config['results_dir'] + config['bias_space_results_file']
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame({key: [] for key in result_keys})
    data_train_str = '/'

    # evaluate bias spaces for the different protected attributes, data_loader, models...
    # skip those experiments where results are already available
    for attr, params in eval_setups_by_attr.items():
        assert 'eval' in params.keys(), "expected key 'eval' in setup for attr %s" % attr
        assert 'defining_terms' in params.keys(), "expected key 'defining_terms' in setup for attr %s" % attr
        assert 'groups' in params.keys(), "expected key 'groups' in setup for attr %s" % attr

        for model in models:
            # determine batch size and pooling choices for current model
            batch_size = 1
            if model in batch_size_lookup.keys():
                batch_size = batch_size_lookup[model]
            pooling_choices = config["pooling"]
            if model in openai_models:
                pooling_choices = ['']

            for pool in pooling_choices:
                for eval_setup in params['eval']:
                    # create filter to determine if this setup has already been evaluated
                    if pool == '':
                        cur_params = [attr, data_train_str, eval_setup['dataset'], model]
                    else:
                        cur_params = [attr, data_train_str, eval_setup['dataset'], model, pool]
                    cur_params_per_key = dict(zip(result_keys[:len(cur_params)], cur_params))
                    result_filter = functools.reduce(lambda a, b: a & b, [(results[key] == val) for key, val in
                                                                          cur_params_per_key.items()])

                    if results.loc[result_filter].empty:
                        # this combination of train and eval setup has not been evaluated yet
                        print("testing %s features on %s" % (attr, eval_setup['dataset']))

                        statistics, pvalues, groups_feature, groups_eval = evaluate_bias_space(params['defining_terms'], params['groups'],
                                                                           eval_setup['dataset'], model, pool,
                                                                           batch_size, config["embedding_dir"],
                                                                           plot_dir=config["plot_dir"],
                                                                           file_prefix=attr,
                                                                           local_dir=eval_setup['local_dir'],
                                                                           sel_groups_eval=eval_setup['groups'],
                                                                           sel_groups_bias_space=eval_setup['groups_pie'])

                        for i, group in enumerate(groups_feature):
                            results.loc[len(results.index)] = [attr, data_train_str, eval_setup['dataset'], model, pool,
                                                               groups_eval[i], group, 0, statistics[i], pvalues[i]]

                    results.to_csv(results_path, index=False)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('bias_space_eval.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('bias_space_eval.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run_bias_space_eval(config)

if __name__ == "__main__":
    main(sys.argv[1:])
