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
from models import CAV
import plotting
import utils


def get_cav_savefile(cav_dir, dataset, model, pooling, file_suffix=None):
    model = model.replace('/','_')
    if file_suffix is not None:
        file_name = cav_dir + ("%s_%s_%s" % (dataset, model, pooling)) + file_suffix + ".pickle"
    else:
        file_name = cav_dir + ("%s_%s_%s" % (dataset, model, pooling)) + ".pickle"
    return file_name

def train_cavs(dataset, model, pooling, batch_size, emb_dir, cav_dir, local_dir=None, sel_groups=None, file_suffix=None):
    file_name = get_cav_savefile(cav_dir, dataset, model, pooling, file_suffix)
    if os.path.isfile(file_name):
        print("cav savefile for %s, %s, %s, %s already exists" % (dataset, model, pooling, file_suffix))
        return
    
    X_train, emb_train, y_train, g_train, X_test, emb_test, y_test, g_test, groups, _, _ = utils.get_dataset_and_embeddings(emb_dir, dataset, model, pooling, batch_size, local_dir, sel_groups=sel_groups)

    if sel_groups is not None:
        print("expecting ", sel_groups)
        assert sel_groups == groups, "expected these groups: %s, instead got: %s" % (sel_groups, groups)

    # datasets with binary group labels provide only 1D labels for two groups; adjust label accordingly
    # in this case, group labels will be learned as single label and are converted to multi-label for evaluation
    if len(groups) == 2 and (g_test.ndim == 1 or g_test.shape[1] == 1):
        groups = [groups[0]+'/'+groups[1]]

    print("fit CAV for %s, %s, %s (%s)" % (dataset, model, pooling, file_suffix))
    print("with groups ", groups)
    
    cav = CAV()
    if len(g_train) == 0:
        # some datasets only have test data, train on test data for cross dataset transfer
        # testing on same dataset should not be done
        cav.fit(emb_test, g_test)
    else:
        cav.fit(emb_train, g_train)
    # g_pred = cav.get_concept_activations(emb_test)
    cavs = cav.get_concept_vectors()

    save_dict = {'cavs': cavs, 'labels': groups}
    with open(file_name, "wb") as handle:
        pickle.dump(save_dict, handle)

def evaluate_cavs(dataset_train, dataset_test, model, pooling, batch_size, emb_dir, cav_dir, plot_dir, local_dir=None, sel_groups_train=None, sel_groups_test=None, file_suffix=None):
    file_name = get_cav_savefile(cav_dir, dataset_train, model, pooling, file_suffix)

    # load CAVs from file, check label consistency
    with open(file_name, "rb") as handle:
        save_dict = pickle.load(handle)
    cavs = save_dict['cavs']
    if sel_groups_train is not None:
        err_msg = "loaded CAVs, but the groups from savefile (%s) do not match the selected groups (%s)" % (save_dict['labels'], sel_groups_train)
        if len(save_dict['labels']) == 1 and len(sel_groups_train) == 2:
            single_label = "%s/%s" % (sel_groups_train[0], sel_groups_train[1])
            assert save_dict['labels'][0] == single_label, err_msg
        else:
            assert save_dict['labels'] == sel_groups_train, err_msg
    else:
        sel_groups_train = save_dict['labels']

    # get the eval dataset + embeddings
    _, _, _, _, X_test, emb_test, y_test, g_test, groups_test, _, _ = utils.get_dataset_and_embeddings(emb_dir, dataset_test, model, pooling, batch_size, local_dir, sel_groups=sel_groups_test)

    # compute concept activations and labels
    pred = np.dot(emb_test, cavs.T)
    g_pred = (pred > 0).astype('int')

    # in cross-dataset transfer evaluations, we might need to test single-labels vs. multi-labels
    # the single-labels are converted to one-hot encoding
    # real valued predictions changed to [-pred, pred]
    if not utils.is1D(g_test) and utils.is1D(g_pred):
        print("convert 1d prediction to onehot")
        g_pred = label2onehot(g_pred, minv=int(np.min(g_test)), maxv=int(np.max(g_test)))
        pred = np.hstack([-pred, pred])
        assert len(sel_groups_train) == 2
    if utils.is1D(g_test) and not utils.is1D(g_pred):
        print("convert 1d test labels to onehot")
        g_test = label2onehot(g_test)
        assert len(groups_test) == 2
    if utils.is1D(g_test) and  utils.is1D(g_pred):
        sel_groups_train = ["%s/%s" % (sel_groups_train[0], sel_groups_train[1])]
        groups_test = ["%s/%s" % (groups_test[0], groups_test[1])]

    # evaluate the concept activations
    # if test and train labels do not exactly match, this can be handled by utils.eval_with_label_match
    # otherwise just compute F1-score and Pearson correlation
    if len(sel_groups_train) != len(groups_test) or save_dict['labels'] != sel_groups_train:
        f1, cav_corr, groups_train_ordered = utils.eval_with_label_match(g_test, g_pred, pred, sel_groups_test, sel_groups_train, average='macro')
    else:
        f1 = f1_score(g_test, g_pred, average='macro')
        groups_train_ordered = groups_test

        if not utils.is1D(pred): # both > 1D
            cav_corr = scipy.stats.pearsonr(pred, np.asarray(g_test))
        else: # both 1D
            cav_corr = scipy.stats.pearsonr(pred.flatten(), g_test.flatten())

    # plot histograms for all test groups (1 or 0) against all predicted concepts, save plot
    model_name = model.replace('/','_')
    plot_dir = '%s/cav_eval' % plot_dir
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if file_suffix is not None:
        plot_savefile = ('%s/%s_%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling, file_suffix))
    else:
        plot_savefile = ('%s/%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling))

    plotting.plot_feature_histogram(pred, g_test, labels=groups_test, features=sel_groups_train, xlabel=dataset_train, ylabel=dataset_test, savefile=plot_savefile)

    return f1, cav_corr, groups_train_ordered, groups_test


def run_cav_training(config):
    # config with training setups (datasets, selected groups...)
    with open(config["cav_train_config"], 'r') as stream:
        training_setups = yaml.safe_load(stream)

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    if not os.path.isdir(config["cav_dir"]):
        os.makedirs(config["cav_dir"])

    # train the CAV models for any combination of LM, pooling and dataset
    for model in models:
        pooling_choices = config["pooling"]
        batch_size = 1
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        if model in openai_models:
            pooling_choices = ['']
        for pool in pooling_choices:
            for setup in training_setups:
                train_cavs(setup['dataset'], model, pool, batch_size, config["embedding_dir"], config["cav_dir"], local_dir=setup['local_dir'], sel_groups=setup['groups'], file_suffix=setup['suffix'])

def run_cav_eval(config):
    # prepare directory where results will be saved
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])

    # config with evaluation setups (datasets, protected attributes and groups...)
    with open(config['eval_config'], 'r') as stream:
        eval_setups_by_attr = yaml.safe_load(stream)

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    result_keys_cav = ["protected_attr", "dataset (train)", "dataset (eval)", "embedder", "pooling", "group (test)", "concept (train)", "F1", "Pearson R", "Pearson p"]

    # read existing results or create new dataframe
    results_cav_path = config['results_dir'] + config['cav_results_file']
    if os.path.isfile(results_cav_path):
        results_cav = pd.read_csv(results_cav_path)
    else:
        results_cav = pd.DataFrame({key: [] for key in result_keys_cav})

    # evaluate CAVs for the different protected attributes, datasets, models...
    # skip those experiments where results are already available
    for attr, eval_setups in eval_setups_by_attr.items():
        for model in models:
            # determine batch size and pooling choices for current model
            batch_size = 1
            if model in batch_size_lookup.keys():
                batch_size = batch_size_lookup[model]
            pooling_choices = config["pooling"]
            if model in openai_models:
                pooling_choices = ['']

            for pool in pooling_choices:
                for train_setup in eval_setups:
                    for eval_setup in eval_setups:
                        # skip 'same dataset' cases where no training data is available
                        if train_setup['dataset'] == eval_setup['dataset'] and eval_setup['dataset'] in ['twitterAAE', 'crows_pairs']:
                            continue  # no training data available for twitterAAE and crowspairs (TODO CV on test data)

                        # create filter to determine if this setup has already been evaluated
                        if pool == '':
                            cur_params = [attr, train_setup['dataset'], eval_setup['dataset'], model]
                        else:
                            cur_params = [attr, train_setup['dataset'], eval_setup['dataset'], model, pool]
                        cur_params_per_key = dict(zip(result_keys_cav[:len(cur_params)], cur_params))
                        result_filter = functools.reduce(lambda a, b: a & b, [(results_cav[key] == val) for key, val in cur_params_per_key.items()])

                        if results_cav.loc[result_filter].empty:
                            # this combination of train and eval setup has not been evaluated yet
                            print("testing %s features on %s" % (train_setup['dataset'], eval_setup['dataset']))

                            # run eval
                            f1, corr, groups_train, groups_eval = evaluate_cavs(train_setup['dataset'], eval_setup['dataset'], model, pool, batch_size, config["embedding_dir"], config["cav_dir"], config["plot_dir"], local_dir=eval_setup['local_dir'], sel_groups_train=train_setup['groups'], sel_groups_test=eval_setup['groups'], file_suffix=train_setup['suffix'])

                            if len(groups_train) == 1:
                                group = groups_train[0]
                                results_cav.loc[len(results_cav.index)] = [attr, train_setup['dataset'],
                                                                           eval_setup['dataset'], model, pool, group,
                                                                           group, f1, corr.statistic, corr.pvalue]
                            else:
                                assert len(groups_train) == len(corr.statistic), ("list of groups and correlation results do not match: %i / %i" % (len(groups_eval_order), len(corr.statistic)))
                                # save results to dataframe (one row per group)
                                for i, group in enumerate(groups_train):
                                    results_cav.loc[len(results_cav.index)] = [attr, train_setup['dataset'],
                                                                               eval_setup['dataset'], model, pool, groups_eval[i],
                                                                               group, f1, corr.statistic[i], corr.pvalue[i]]

            results_cav.to_csv(results_cav_path, index=False)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('cav_train_eval.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('cav_train_eval.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run_cav_training(config)
    run_cav_eval(config)

if __name__ == "__main__":
    main(sys.argv[1:])
