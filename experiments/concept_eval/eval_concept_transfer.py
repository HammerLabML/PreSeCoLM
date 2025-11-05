import os
import sys
import json
import yaml
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
from evaluate_concepts import train_eval_one_split, get_parameter_sets

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


def evaluate(wrapper: models.ClfWrapper, emb_test: np.ndarray, y_test: np.ndarray,
             pred_test_ids: list) -> list:
    # evaluate PR-AUC per group
    concepts = wrapper.predict(emb_test)
    aucs = []
    for (pred_id, test_id) in pred_test_ids:
        precision, recall, thresh = precision_recall_curve(y_test[:, test_id], concepts[:, pred_id])
        aucs.append(auc(recall, precision))

    return aucs


def train_evaluate_indomain(dataset_train, clf_class, cur_clf_params: dict, cur_wrapper_params: dict,
                            epochs: int) -> (models.ClfWrapper, float, int):
    clf_params = copy.deepcopy(cur_clf_params)
    if 'hidden_size_factor' in cur_clf_params.keys():
        assert (0 < clf_params['hidden_size_factor'] <= 1)
        clf_params['hidden_size'] = int(clf_params['input_size'] * clf_params['hidden_size_factor'])
        clf_params.pop('hidden_size_factor', None)

    # get training and validation split from the training dataset
    if len(dataset_train.splits) == 1:  # test set only
        split_name = dataset_train.splits[0]
        _, emb_train, _, g_train, _, gw = dataset_train.get_split(split_name)
        emb_train, emb_test, g_train, g_test = train_test_split(emb_train, g_train, test_size=0.2)
        emb_train, emb_val, g_train, g_val = train_test_split(emb_train, g_train, test_size=0.1)
    else:  # presumably train, dev, test split
        _, emb_train, _, g_train, _, gw = dataset_train.get_split('train')
        _, emb_val, _, g_val, _, _ = dataset_train.get_split('dev')
        _, emb_test, _, g_test, _, _ = dataset_train.get_split('test')

    # fit and predict concepts
    clf = clf_class(**clf_params)
    wrapper = models.ClfWrapper(clf, **cur_wrapper_params, class_weights=gw)
    epochs = wrapper.fit_early_stopping(X_train=emb_train, y_train=g_train, X_val=emb_val, y_val=g_val,
                                        max_epochs=epochs, delta=0.01, patience=10)

    dummy_id_lookup = [(idx, idx) for idx in range(len(dataset_train.group_names))]
    aucs = evaluate(wrapper, emb_test, g_test, dummy_id_lookup)

    return wrapper, aucs, epochs


def get_standardized_label(label_match: dict, label: str) -> str:
    for attr, label_dict in label_match.items():
        for standard_lbl, labels in label_dict.items():
            if label in labels:
                return standard_lbl


def get_matching_group_ids(label_match_lookup: dict, dataset_train: data_loader.CustomDataset,
                            dataset_test: data_loader.CustomDataset) -> (list, list, list):
    # standardize dataset-specific group label and find shared label indices
    eval_groups_s = [get_standardized_label(label_match_lookup, lbl) for lbl in dataset_test.group_names]
    train_groups_s = [get_standardized_label(label_match_lookup, lbl) for lbl in dataset_train.group_names]
    shared_lbl = [lbl for lbl in eval_groups_s if lbl in train_groups_s and lbl is not None]

    eval_ids = [eval_groups_s.index(lbl) for lbl in shared_lbl]
    train_ids = [train_groups_s.index(lbl) for lbl in shared_lbl]

    pred_test_id = list(zip(train_ids, eval_ids))
    return shared_lbl, pred_test_id


def get_classifier_param_sets(clf_param_dict, input_size, output_size):
    # create list with classifier/ parameter configurations
    # add input and output size (matching the embedding size to clf parameters)
    clf_parameters = copy.deepcopy(clf_param_dict)

    for key in clf_param_dict:
        if key != 'wrapper':
            print("set input and output dim for clf: " + key)
            clf_parameters[key]['input_size'] = input_size
            clf_parameters[key]['output_size'] = output_size
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

    return clf_parameter_sets


def are_results_missing(results, key_param_dict):
    if key_param_dict['pooling'] == '':
        key_param_dict.pop('pooling', None)
    result_filter = functools.reduce(lambda a, b: a & b,
                                     [(results[key] == val) for key, val in key_param_dict.items()])
    return results.loc[result_filter].empty


def eval_all_clf_choices(results: pd.DataFrame, results_path: str, dataset_train: data_loader.CustomDataset,
                         dataset_test_setups: list, model_name: str, pooling: str, batch_size: int, emb_dir: str,
                         clf_param_dict: dict, max_epochs: int, emb_dim: int, label_match_lookup: dict):
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """
    model_type, model_architecture = utils.get_model_type_architecture(model_name)
    clf_parameter_sets = get_classifier_param_sets(clf_param_dict, input_size=emb_dim, output_size=dataset_train.n_groups)
    classifier_choices = [key for key in clf_parameter_sets.keys() if key != 'wrapper']

    # find test setups where no results exist so far
    matching_test_setups = []
    for test_setup in dataset_test_setups:
        # load test dataset and determine shared labels
        dataset_test = utils.get_dataset_with_embeddings(emb_dir, test_setup['name'], model_name, pooling, batch_size,
                                                         test_setup['local_dir'])
        shared_lbl, pred_test_id_lookup = get_matching_group_ids(label_match_lookup, dataset_train, dataset_test)

        # check that there are labels for transfer eval
        if len(shared_lbl) > 1:

            # check if some results are missing
            results_missing = False
            for clf in classifier_choices:
                for clf_params in clf_parameter_sets[clf]:
                    for wrapper_params in clf_parameter_sets['wrapper']:
                        key_param_dict = {'dataset train': dataset_train.name, 'dataset test': dataset_test.name,
                                          'model': model_name, 'pooling': pooling}
                        if are_results_missing(results, key_param_dict):
                            results_missing = True
            if results_missing:
                setup = test_setup.copy()
                setup['dataset'] = dataset_test
                setup['shared_lbl'] = shared_lbl
                setup['pred_test_id_lookup'] = pred_test_id_lookup
                matching_test_setups.append(setup)

    if len(matching_test_setups) == 0:
        print("nothing to do for %s %s %s" % (dataset_train.name, model_name, pooling))
        return

    for clf in classifier_choices:
        for clf_params in clf_parameter_sets[clf]:
            for wrapper_params in clf_parameter_sets['wrapper']:

                # train the classifier
                try:  # Salsa might fail
                    clf_wrapper, aucs_indomain, ep = train_evaluate_indomain(dataset_train,
                                                                             clf_head_lookup[clf],
                                                                             clf_params,
                                                                             wrapper_params, max_epochs)
                except ValueError as error:
                    print("learning failed for %s on %s  (ValueError)" % (dataset_train.name, model_name))
                    print(error)
                    continue
                except RuntimeError as error:
                    print("learning failed for %s on %s (RunetimeError)" % (dataset_train.name, model_name))
                    print(error)
                    continue

                # for logging
                hidden_size = -1
                if 'hidden_size_factor' in clf_params.keys():
                    hidden_size = clf_params['hidden_size_factor']
                optim = list(optimizer_lookup.keys())[
                    list(optimizer_lookup.values()).index(wrapper_params['optimizer'])]
                loss_fct = list(criterion_lookup.keys())[
                    list(criterion_lookup.values()).index(wrapper_params['criterion'])]

                # log results in-domain
                aucs_indomain_mean = np.mean(aucs_indomain)
                results.loc[len(results.index)] = [dataset_train.name, dataset_train.name, model_name, model_type,
                                                   model_architecture, pooling, clf, hidden_size, emb_dim, optim,
                                                   wrapper_params['lr'], loss_fct, "mean", aucs_indomain_mean]
                for i, group in enumerate(dataset_train.group_names):
                    results.loc[len(results.index)] = [dataset_train.name, dataset_train.name, model_name, model_type,
                                                       model_architecture,
                                                       pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                       loss_fct, group, aucs_indomain[i]]

                # test the classifier on all possible transfer datasets
                for test_setup in matching_test_setups:
                    key_param_dict = {'dataset train': dataset_train.name, 'dataset test': test_setup['name'],
                                      'model': model_name, 'pooling': pooling}
                    if are_results_missing(results, key_param_dict):
                        print("run experiment for %s -> %s %s %s" % (dataset_train.name, test_setup['name'],
                                                                     model_name, pooling))
                        dataset_test = test_setup['dataset']
                        shared_lbl = test_setup['shared_lbl']
                        pred_test_id_lookup = test_setup['pred_test_id_lookup']

                        # load test dataset and get test split
                        test_split_name = 'test' if 'test' len(dataset_test.splits) > 1 else dataset_test.splits[0]
                        _, emb_test_t, _, g_test_t, _, _ = dataset_test.get_split(test_split_name)
                        aucs_transfer = evaluate(clf_wrapper, emb_test_t, g_test_t, pred_test_id_lookup)

                        # log results for transfer
                        aucs_transfer_mean = np.mean(aucs_transfer)
                        results.loc[len(results.index)] = [dataset_train.name, dataset_test.name, model_name, model_type,
                                                           model_architecture, pooling, clf, hidden_size, emb_dim, optim,
                                                           wrapper_params['lr'], loss_fct, "mean", aucs_transfer_mean]

                        print(shared_lbl)
                        print(pred_test_id_lookup)
                        print(aucs_transfer)
                        # results (one row for each protected group)
                        for i, group in enumerate(shared_lbl):
                            results.loc[len(results.index)] = [dataset_train.name, dataset_test.name, model_name, model_type, model_architecture,
                                                               pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                               loss_fct, group, aucs_transfer[i]]

                print("save results for setup: %s, %s, %s, %s" % (dataset_train.name, model_name, pooling, clf))
                results.to_csv(results_path, index=False)

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
    result_keys = ["dataset train", "dataset test", "model", "model type", "architecture", "pooling", "classifier",
                   "clf hidden size factor", "emb size", "optimizer", "lr", "loss",
                   "group", "PR-AUC"]
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame({key: [] for key in result_keys})

    with open(config['label_match_config'], 'r') as ff:
        label_match_config = yaml.safe_load(ff)

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
            for dataset_setup_train in config['datasets']:
                dataset_train = utils.get_dataset_with_embeddings(config['embedding_dir'], dataset_setup_train['name'],
                                                                  model, pool, batch_size,
                                                                  dataset_setup_train['local_dir'])
                test_setups = [setup for setup in config['datasets'] if setup['name'] != dataset_setup_train['name']]
                eval_all_clf_choices(results, results_path, dataset_train, test_setups, model, pool, batch_size,
                                     config['embedding_dir'], config['classifier'], config['max_epochs'], lm_emb_size,
                                     label_match_config)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('eval_concept_transfer.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('eval_concept_transfer.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
