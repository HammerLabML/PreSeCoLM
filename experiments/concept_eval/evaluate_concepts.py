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


def train_eval_one_split(emb_train: np.ndarray, y_train: np.ndarray,  emb_val: np.ndarray, y_val: np.ndarray,
                         emb_test: np.ndarray, y_test: np.ndarray, groups_names: list,
                         clf_class, cur_clf_params: dict, cur_wrapper_params: dict,
                         concept_weights: np.ndarray, epochs: int) -> (float, float, float):
    # here y_* are the concept labels not class labels (always multi-label)

    clf_params = copy.deepcopy(cur_clf_params)
    if 'hidden_size_factor' in cur_clf_params.keys():
        assert (0 < clf_params['hidden_size_factor'] <= 1)
        clf_params['hidden_size'] = int(clf_params['input_size'] * clf_params['hidden_size_factor'])
        clf_params.pop('hidden_size_factor', None)

    # fit and predict concepts
    clf = clf_class(**clf_params)
    wrapper = models.ClfWrapper(clf, **cur_wrapper_params, class_weights=concept_weights)
    epochs = wrapper.fit_early_stopping(X_train=emb_train, y_train=y_train, X_val=emb_val, y_val=y_val,
                                        max_epochs=epochs, delta=0.01, patience=10)
    concepts = wrapper.predict(emb_test)

    # compute Pearson correlation for each group concept
    corrs = []
    pvalues = []
    aucs = []
    for gid, group in enumerate(groups_names):
        r, p = scipy.stats.pearsonr(concepts[:, gid], y_test[:, gid])
        precision, recall, thresh = precision_recall_curve(y_test[:, gid], concepts[:, gid])
        aucs.append(auc(recall, precision))
        corrs.append(r)
        pvalues.append(p)

    print("PR-AUC: ", aucs)

    return corrs, pvalues, aucs, epochs


def eval_cv(dataset: data_loader.CustomDataset, group_names: list, clf_class: torch.nn.Module,
            cur_clf_params: dict, cur_wrapper_params: dict, max_epochs: int) -> (float, float, float):
    rvalues = []
    pvalues = []
    aucs = []
    epochs = []
    for fold_id in range(dataset.n_folds):
        data_dict = dataset.get_cv_split(fold_id)
        X_train, emb_train, _, g_train, _, gw = data_dict['train']
        X_test, emb_test, _, g_test, _, _ = data_dict['test']

        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, group_names, g_train)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, group_names, g_test)
        emb_train, emb_val, g_train, g_val = train_test_split(emb_train, g_train, test_size=0.1)

        (cur_corrs, cur_pval, cur_auc, ep) = train_eval_one_split(emb_train, g_train, emb_val, g_val,
                                                                  emb_test, g_test, group_names, clf_class,
                                                                  cur_clf_params, cur_wrapper_params, gw,
                                                                  max_epochs)
        rvalues.append(cur_corrs)
        pvalues.append(cur_pval)
        aucs.append(cur_auc)
        epochs.append(ep)

    corrs = np.vstack(rvalues)
    ps = np.vstack(pvalues)

    return np.mean(corrs, axis=0), np.mean(ps, axis=0), np.mean(aucs, axis=0), epochs


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
                         emb_dim: int, local_dir: str = None) -> pd.DataFrame:
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """

    model_type, model_architecture = utils.get_model_type_architecture(model_name)
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset_name, model_name, pooling, batch_size, local_dir)
    sel_groups = dataset.group_names
    n_protected_concepts = len(sel_groups)
    print(sel_groups)
    print("run experiment for dataset %s" % (dataset_name))

    # we apply CV for (small) datasets that do not provide a train-test split
    use_cv = (len(dataset.splits) == 1)

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

    if not use_cv:
        X_train, emb_train, _, g_train, _, gw = dataset.get_split('train')
        X_dev, emb_dev, _, g_dev, _, _ = dataset.get_split('dev')
        X_test, emb_test, _, g_test, _, gw_test = dataset.get_split('test')
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_train)
        g_dev, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_dev)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)
    else:
        _, _, _, _, _, gw_test = dataset.get_split(dataset.splits[0])

    classifier_choices = [key for key in clf_parameters.keys() if key != 'wrapper']
    for clf in classifier_choices:
        for clf_params in clf_parameter_sets[clf]:
            for wrapper_params in clf_parameter_sets['wrapper']:
                try:  # salsa might fail
                    if use_cv:
                        corr, pval, aucs, ep = eval_cv(dataset, sel_groups, clf_head_lookup[clf],
                                                       clf_params, wrapper_params, max_epochs)
                    else:
                        corr, pval, aucs, ep = train_eval_one_split(emb_train, g_train, emb_dev, g_dev, emb_test,
                                                                    g_test, sel_groups, clf_head_lookup[clf],
                                                                    clf_params, wrapper_params, gw, max_epochs)

                except ValueError as error:
                    print("learning failed for %s on %s  (ValueError)" % (model_name, dataset_name))
                    print(error)
                    continue
                except RuntimeError as error:
                    print("learning failed for %s on %s (RunetimeError)" % (model_name, dataset_name))
                    print(error)
                    continue

                hidden_size = -1
                if 'hidden_size_factor' in clf_params.keys():
                    hidden_size = clf_params['hidden_size_factor']
                optim = list(optimizer_lookup.keys())[list(optimizer_lookup.values()).index(wrapper_params['optimizer'])]
                loss_fct = list(criterion_lookup.keys())[list(criterion_lookup.values()).index(wrapper_params['criterion'])]
                emb_dim = dataset.data_preprocessed[dataset.splits[0]].shape[1]

                # results (mean weighted/ unweighted by group support)
                aucs_wm = np.sum([aucs[i]*gw_test[i] for i in range(len(aucs))]) / np.sum(gw_test)
                aucs_m = np.mean(aucs)
                corr_wm = np.sum([corr[i]*gw_test[i] for i in range(len(corr))]) / np.sum(gw_test)
                corr_m = np.mean(corr)
                pval_wm = np.sum([pval[i] * gw_test[i] for i in range(len(pval))]) / np.sum(gw_test)
                pval_m = np.mean(pval)
                results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture,
                                                   pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                   loss_fct, "mean", corr_m, pval_m, aucs_m, ep]
                results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture,
                                                   pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                   loss_fct, "weighted mean", corr_wm, pval_wm, aucs_wm, ep]

                # results (one row for each protected group)
                for i, group in enumerate(sel_groups):
                    results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture,
                                                       pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                       loss_fct, group, corr[i], pval[i], aucs[i], ep]

    return results


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
                   "group", "Pearson R", "pvalue", "PR-AUC", "Epochs"]
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
                # check if results exist and this setup can be skipped
                if pool == '':
                    cur_params_per_key = {'dataset': dataset_setup['name'], 'model': model}
                else:
                    cur_params_per_key = {'dataset': dataset_setup['name'], 'model': model,
                                          'pooling': pool}
                result_filter = functools.reduce(lambda a, b: a & b,
                                                 [(results[key] == val) for key, val in cur_params_per_key.items()])

                if results.loc[result_filter].empty:
                    print("run experiment for %s %s %s" % (dataset_setup['name'], model, pool))
                    results = eval_all_clf_choices(results, dataset_setup['name'], model, pool, batch_size,
                                                   config['embedding_dir'], config['classifier'], config['max_epochs'],
                                                   lm_emb_size, local_dir=dataset_setup['local_dir'])
                    print("save results for setup: %s, %s, %s" % (dataset_setup['name'], model, pool))
                    results.to_csv(results_path, index=False)


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
