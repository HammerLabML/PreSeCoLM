import os
import sys
import json
import pickle
import pandas as pd
import scipy

import functools
import itertools
import getopt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from datetime import datetime
from embedding import BertHuggingface

import copy
from salsa.SaLSA import SaLSA

# local imports
from evaluate_concepts import train_eval_one_split

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import data_loader
import models


optimizer_lookup = {'Salsa': SaLSA, 'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW, 'Adamax': torch.optim.Adamax,
                    'Adadelta': torch.optim.Adadelta, 'Adagrad': torch.optim.Adagrad, 'SparseAdam': torch.optim.SparseAdam, 'ASGD': torch.optim.ASGD,
                    'SGD': torch.optim.SGD, 'LBFGS': torch.optim.LBFGS, 'RAdam': torch.optim.RAdam, 'NAdam': torch.optim.NAdam, 'Rprop': torch.optim.Rprop}
criterion_lookup = {'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss, 'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss, 'L1Loss': torch.nn.L1Loss, 'CrossEntropyLoss': torch.nn.CrossEntropyLoss, 'GaussianNLLLoss': torch.nn.GaussianNLLLoss,
              'MSELoss': torch.nn.MSELoss, 'CTCLoss': torch.nn.CTCLoss, 'NLLLoss': torch.nn.NLLLoss, 'PoissonNLLLoss': torch.nn.PoissonNLLLoss, 'KLDivLoss': torch.nn.KLDivLoss, 'BCELoss': torch.nn.BCELoss, 'MarginRankingLoss': torch.nn.MarginRankingLoss,
              'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss, 'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss, 'HuberLoss': torch.nn.HuberLoss, 'SmoothL1Loss': torch.nn.SmoothL1Loss, 'SoftMarginLoss': torch.nn.SoftMarginLoss,
              'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss, 'MultiMarginLoss': torch.nn.MultiMarginLoss, 'TripletMarginLoss': torch.nn.TripletMarginLoss, 'TripletMarginWithDistanceLoss': torch.nn.TripletMarginWithDistanceLoss}
clf_head_lookup = {'MLP2': models.MLP2Layer, 'linear': models.LinearClassifier, 'MLP3': models.MLP3Layer}


def sample_to_ratio(X, y, ratio, random_state=42):
    assert 0 < ratio < 1
    np.random.seed(random_state)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n = n_pos+n_neg

    cur_ratio = n_pos / n

    if cur_ratio < ratio:
        # downsample negative samples
        n_pos_target = n_pos
        n_target = int(n_pos_target/ratio)
        n_neg_target = n_target - n_pos_target
    else:
        # downsample positive samples
        n_neg_target = n_neg
        n_target = int(n_neg_target/(1-ratio))
        n_pos_target = n_target - n_neg_target

    print("target samples (n/pos/neg):", n_target, n_pos_target, n_neg_target)

    pos_sample = np.random.choice(pos_idx, n_pos_target, replace=False)
    neg_sample = np.random.choice(neg_idx, n_neg_target, replace=False)

    selected_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(selected_idx)

    return X[selected_idx], y[selected_idx].reshape(-1, 1)


def eval_cv(dataset: data_loader.CustomDataset, group: str, group_ratio: float, clf_class: torch.nn.Module,
            cur_clf_params: dict, cur_wrapper_params: dict, max_epochs: int) -> (float, float, float):
    rvalues = []
    pvalues = []
    aucs = []
    epochs = []
    group_weights = []
    for fold_id in range(dataset.n_folds):
        data_dict = dataset.get_cv_split(fold_id)
        X_train, emb_train, _, g_train, _, gw = data_dict['train']
        X_test, emb_test, _, g_test, _, _ = data_dict['test']

        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, [group], g_train)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, [group], g_test)

        # sample according to group ratio
        emb_train, g_train = sample_to_ratio(emb_train, g_train, ratio=group_ratio)
        cur_clf_params['output_size'] = 1
        gw = [len(g_train)/np.sum(g_train)]

        # train-val split
        emb_train, emb_val, g_train, g_val = train_test_split(emb_train, g_train, test_size=0.1)

        (cur_corrs, cur_pval, cur_auc, ep) = train_eval_one_split(emb_train, g_train, emb_val, g_val,
                                                                  emb_test, g_test, [group], clf_class,
                                                                  cur_clf_params, cur_wrapper_params, gw,
                                                                  max_epochs)
        if np.isnan(cur_corrs).any():
            # auc would be 0.5 in that case, set to nan, so it doesn't influence the mean result
            cur_auc = [np.nan if np.isnan(r) else a for a, r in zip(cur_auc, cur_corrs)]

        rvalues.append(cur_corrs)
        pvalues.append(cur_pval)
        aucs.append(cur_auc)
        epochs.append(ep)
        group_weights.append(gw)

    corrs = np.vstack(rvalues)
    ps = np.vstack(pvalues)

    print(corrs)
    print(np.nanmean(corrs))

    return np.nanmean(corrs, axis=0), np.nanmean(ps, axis=0), np.nanmean(aucs, axis=0), epochs, np.mean(group_weights, axis=0)


def get_parameter_sets(choices_per_param: dict):
    params_fix = []
    params_choices = []
    for param, values in choices_per_param.items():
        if type(values) == list:
            params_choices.append(param)
        else:
            params_fix.append(param)

    s = [list(zip([param] * len(choices_per_param[param]), choices_per_param[param])) for param in params_choices]
    permutations = list(itertools.product(*s))
    parameter_sets = [{key: val for (key, val) in elem} for elem in permutations]
    for pset in parameter_sets:
        pset.update({param: choices_per_param[param] for param in params_fix})

    return parameter_sets


def eval_all_clf_choices(results: pd.DataFrame, dataset_name: str, model_name: str, pooling: str,
                         batch_size: int, emb_dir: str, clf_param_dict: dict, max_epochs: int,
                         emb_dim: int, sel_group: str, group_ratio: float, results_path, local_dir: str = None):
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """

    model_type, model_architecture = utils.get_model_type_architecture(model_name)
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset_name, model_name, pooling, batch_size, local_dir)
    print(sel_group)

    if sel_group not in dataset.group_names:
        return

    print("run experiment for dataset %s" % dataset_name)

    # we apply CV for (small) datasets that do not provide a train-test split

    # create list with classifier/ parameter configurations
    # add input and output size (matching the embedding size to clf parameters)
    clf_parameters = copy.deepcopy(clf_param_dict)

    for key in clf_param_dict:
        if key != 'wrapper':
            print("set input and output dim for clf: " + key)
            clf_parameters[key]['input_size'] = emb_dim
            clf_parameters[key]['output_size'] = dataset.n_groups
    clf_parameters['wrapper']['optimizer'] = [optimizer_lookup[optim] for optim in
                                              clf_param_dict['wrapper']['optimizer']]
    # concepts are always multi-label -> loss
    clf_parameters['wrapper']['criterion'] = [criterion_lookup[loss] for loss in
                                              clf_param_dict['wrapper']['criterion']['multi-label']]

    # get all possible set of parameters for the classifiers
    clf_parameter_sets = {}
    for key, param_dict in clf_parameters.items():
        clf_parameter_sets[key] = get_parameter_sets(param_dict)
    print("clf/ wrapper parameter sets:")
    print(clf_parameter_sets)

    classifier_choices = [key for key in clf_parameters.keys() if key != 'wrapper']
    for clf in classifier_choices:
        for clf_params in clf_parameter_sets[clf]:
            for wrapper_params in clf_parameter_sets['wrapper']:
                try:  # salsa might fail
                    corr, pval, aucs, ep, weights = eval_cv(dataset, sel_group, group_ratio, clf_head_lookup[clf],
                                                            clf_params, wrapper_params, max_epochs)

                #except ValueError as error:
                #    print("learning failed for %s on %s  (ValueError)" % (model_name, dataset_name))
                #    print(error)
                #    continue
                except RuntimeError as error:
                    print("learning failed for %s on %s (RuntimeError)" % (model_name, dataset_name))
                    print(error)
                    continue

                hidden_size = -1
                if 'hidden_size_factor' in clf_params.keys():
                    hidden_size = clf_params['hidden_size_factor']
                optim = list(optimizer_lookup.keys())[list(optimizer_lookup.values()).index(wrapper_params['optimizer'])]
                loss_fct = list(criterion_lookup.keys())[list(criterion_lookup.values()).index(wrapper_params['criterion'])]
                emb_dim = dataset.data_preprocessed[dataset.splits[0]].shape[1]

                results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture,
                                                   pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                   loss_fct, sel_group, group_ratio, corr[0], pval[0], aucs[0], str(ep)]

    print("save results for setup: %s, %s, %s" % (dataset_name, model_name, sel_group))
    results.to_csv(results_path, index=False)


def run(config):

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    model_names = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    # prepare directory where results will be saved
    results_path = config['results_path']
    results_dir = results_path.replace(results_path.split('/')[-1],'')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # results for performance
    result_keys = ["dataset", "model", "model type", "architecture", "pooling", "classifier",
                   "clf hidden size factor", "emb size", "optimizer", "lr", "loss",
                   "group", "ratio", "Pearson R", "pvalue", "PR-AUC", "Epochs"]
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
        print(results)
    else:
        results = pd.DataFrame({key: [] for key in result_keys})

    # evaluate baseline classifier performance on different LMs, pooling choices, datasets
    for model in model_names:
        # load model once to get the embedding dim, but don't block the vram
        if model in huggingface_models:
            lm = BertHuggingface(2, model_name=model, batch_size=1)
            lm_emb_size = lm.model.config.hidden_size
            lm.model = lm.model.to('cpu')
            del lm
        else:  # openai model
            if model == 'text-embedding-3-small':
                lm_emb_size = 1536
            elif model == 'text-embedding-3-large':
                lm_emb_size = 3072
            else:
                print("could not set emb size for model %s, skip this" % model)
                continue

        # set batch size and pooling choices
        pooling_choices = config["pooling"]
        batch_size = 1
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        if model in openai_models:
            pooling_choices = ['']

        for pool in pooling_choices:
            for dataset_setup in config['datasets']:
                for ratio in config['group_ratios']:
                    for group in config['sel_groups']:
                        # check if results exist and this setup can be skipped
                        if pool == '':
                            cur_params_per_key = {'dataset': dataset_setup['name'], 'model': model,
                                                  'group': group, 'ratio': ratio}
                        else:
                            cur_params_per_key = {'dataset': dataset_setup['name'], 'model': model,
                                                  'pooling': pool, 'group': group, 'ratio': ratio}
                        result_filter = functools.reduce(lambda a, b: a & b,
                                                         [(results[key] == val) for key, val in cur_params_per_key.items()])

                        if results.loc[result_filter].empty:
                            print("run experiment for %s %s %s" % (dataset_setup['name'], model, pool))
                            eval_all_clf_choices(results, dataset_setup['name'], model, pool, batch_size,
                                                 config['embedding_dir'], config['classifier'], config['max_epochs'],
                                                 lm_emb_size, group, ratio, results_path,
                                                 local_dir=dataset_setup['local_dir'])



def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('evaluate_concepts.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluate_concepts.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
