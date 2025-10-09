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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import utils

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import data_loader
import models

optimizer_lookup = {'Salsa': SaLSA, 'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam,
                    'AdamW': torch.optim.AdamW, 'Adamax': torch.optim.Adamax,
                    'Adadelta': torch.optim.Adadelta, 'Adagrad': torch.optim.Adagrad,
                    'SparseAdam': torch.optim.SparseAdam, 'ASGD': torch.optim.ASGD,
                    'SGD': torch.optim.SGD, 'LBFGS': torch.optim.LBFGS, 'RAdam': torch.optim.RAdam,
                    'NAdam': torch.optim.NAdam, 'Rprop': torch.optim.Rprop}
criterion_lookup = {'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss,
                    'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss, 'L1Loss': torch.nn.L1Loss,
                    'CrossEntropyLoss': torch.nn.CrossEntropyLoss, 'GaussianNLLLoss': torch.nn.GaussianNLLLoss,
                    'MSELoss': torch.nn.MSELoss, 'CTCLoss': torch.nn.CTCLoss, 'NLLLoss': torch.nn.NLLLoss,
                    'PoissonNLLLoss': torch.nn.PoissonNLLLoss, 'KLDivLoss': torch.nn.KLDivLoss,
                    'BCELoss': torch.nn.BCELoss, 'MarginRankingLoss': torch.nn.MarginRankingLoss,
                    'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss,
                    'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss, 'HuberLoss': torch.nn.HuberLoss,
                    'SmoothL1Loss': torch.nn.SmoothL1Loss, 'SoftMarginLoss': torch.nn.SoftMarginLoss,
                    'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss, 'MultiMarginLoss': torch.nn.MultiMarginLoss,
                    'TripletMarginLoss': torch.nn.TripletMarginLoss,
                    'TripletMarginWithDistanceLoss': torch.nn.TripletMarginWithDistanceLoss}
clf_head_lookup = {'MLP2': models.MLP2Layer, 'linear': models.LinearClassifier, 'MLP3': models.MLP3Layer}


def get_cbm_savefile(cbm_dir, dataset, model, pooling, file_suffix=None):
    model = model.replace('/', '_')
    if file_suffix is not None:
        checkpoint_file = cbm_dir + ("%s_%s_%s_%s" % (dataset, model, pooling, file_suffix))
        params_file = cbm_dir + ("%s_%s_%s.pickle" % (dataset, model, file_suffix))
    else:
        checkpoint_file = cbm_dir + ("%s_%s_%s" % (dataset, model, pooling))
        params_file = cbm_dir + ("%s_%s.pickle" % (dataset, model))

    # avoid unnecessary underscores
    params_file = params_file.replace('__', '_').replace('_.pickle', '.pickle')
    checkpoint_file = checkpoint_file.replace('__', '_')
    if checkpoint_file.endswith('_'):
        checkpoint_file = checkpoint_file[:-1]

    return checkpoint_file, params_file


def train_model(emb_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
                emb_val: np.ndarray, y_val: np.ndarray, g_val: np.ndarray,
                emb_test: np.ndarray, y_test: np.ndarray, g_test: np.ndarray,
                checkpoint_file: str, params_file: str, cur_clf_params: dict, cur_wrapper_params: dict,
                class_weights: np.ndarray, concept_weights: np.ndarray,
                epochs: int, multi_label: bool) -> (float, float, float):
    clf_params = copy.deepcopy(cur_clf_params)
    wrapper_params = copy.deepcopy(cur_wrapper_params)

    if os.path.isfile(checkpoint_file) and os.path.isfile(params_file):
        print("cbm checkpoint and parameter file %s, %s already exist" % (checkpoint_file, params_file))
        return
    else:
        print("checkpoint and/ or parameter file not found: %s, %s" % (checkpoint_file, params_file))

    # set clf input size
    n_concepts = clf_params['n_concepts_protec'] + clf_params['n_concepts_unsup']
    clf_params['input_size'] = emb_train.shape[1]

    wrapper_params['optimizer'] = optimizer_lookup[wrapper_params['optimizer']]
    if multi_label:
        wrapper_params['criterion'] = criterion_lookup[wrapper_params['criterion']['multi-label']]
    else:
        wrapper_params['criterion'] = criterion_lookup[wrapper_params['criterion']['single-label']]
    clf_class = clf_head_lookup[wrapper_params['clf']]
    wrapper_params.pop('clf', None)

    if 'hidden_size_factor' in cur_clf_params.keys():
        assert (0 < clf_params['hidden_size_factor'] <= 1)
        if clf_class == models.MLP3Layer:
            # got 2 hidden layers
            clf_params['hidden_size'] = int(n_concepts * clf_params['hidden_size_factor'])
            clf_params['hidden_size2'] = int(clf_params['hidden_size'] * clf_params['hidden_size_factor'])
        else:
            clf_params['hidden_size'] = int(n_concepts * clf_params['hidden_size_factor'])
        clf_params.pop('hidden_size_factor', None)


    if (y_train == -1).all():
        # we only need the concepts, so we can train on data without class labels
        y_train = g_train.astype(int)
        y_val = g_val.astype(int)
        y_test = g_test.astype(int)
        if multi_label:
            clf_params['output_size'] = y_test.shape[1]
        else:
            clf_params['output_size'] = 2

    if not multi_label and y_train.ndim > 1:
        y_train = np.squeeze(y_train[:, 0])
        y_val = np.squeeze(y_val[:, 0])
        y_test = np.squeeze(y_test[:, 0])

    print(clf_params)
    print(wrapper_params)

    cbm = models.CBM(**clf_params)
    cbmWrapper = models.CBMWrapper(cbm, **wrapper_params, class_weights=class_weights,
                                   concept_criterion=torch.nn.BCEWithLogitsLoss,
                                   concept_weights=concept_weights)
    epochs = cbmWrapper.fit_early_stopping(X_train=emb_train, y_train=y_train, c_train=g_train,
                                           X_val=emb_val, y_val=y_val, c_val=g_val,
                                           max_epochs=epochs, delta=0.01, patience=10)

    for (split, X, y, g) in [('train', emb_train, y_train, g_train), ('val', emb_val, y_val, g_val),
                             ('test', emb_test, y_test, g_test)]:
        pred, concepts = cbmWrapper.predict(X)

        if multi_label:
            y_pred = (pred > 0.5).astype('int')
        else:
            y_pred = np.argmax(pred, axis=1)
        g_pred = (concepts > 0.5).astype('int')

        f1_performance = f1_score(y, y_pred, average='macro')
        f1_concept = f1_score(g, g_pred, average='macro')
        print("[%s F1 score] performance : %.2f, concepts: %.2f" % (split, f1_performance, f1_concept))

    cbm.to_cpu()

    torch.save(cbm.state_dict(), checkpoint_file)
    model_params = {'clf': clf_params, 'wrapper': wrapper_params}
    with open(params_file, 'wb') as handle:
        pickle.dump(model_params, handle)

    del cbm


def training_wrapper(checkpoint_dir: pd.DataFrame, dataset_name: str, model_name: str, pooling: str,
                         batch_size: int, emb_dir: str, clf_parameters: dict, wrapper_parameters: dict,
                         max_epochs: int, emb_dim: int, local_dir: str = None):
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset_name, model_name, pooling, batch_size, local_dir)
    sel_groups = dataset.group_names
    n_protected_concepts = len(sel_groups)
    print(sel_groups)
    print("train CBM for dataset %s, multi_label=%i" % (dataset_name, dataset.multi_label))

    # we apply CV for (small) datasets that do not provide a train-test split
    use_cv = (len(dataset.splits) == 1)

    # set clf and wrapper parameters according to config
    print(clf_parameters)
    clf_parameters['n_concepts_protec'] = n_protected_concepts
    clf_parameters['output_size'] = dataset.n_classes
    if clf_parameters['n_concepts_unsup'] == -1:
        clf_parameters['n_concepts_unsup'] = emb_dim - clf_parameters['n_concepts_protec']
    n_concepts = clf_parameters['n_concepts_protec'] + clf_parameters['n_concepts_unsup']

    print("clf parameters:")
    print(clf_parameters)

    assert n_concepts <= emb_dim

    checkpoint_file, params_file = get_cbm_savefile(checkpoint_dir, dataset_name, model_name, pooling, file_suffix=None)

    if use_cv:
        _, emb_train, y_train, g_train, cw, gw = dataset.get_split(dataset.splits[0])
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_train)
        emb_train, emb_dev, y_train, y_dev, g_train, g_dev = train_test_split(emb_train, y_train, g_train,
                                                                              test_size=0.1)
        # workaround for small datasets; use as much training data as possible, bc these are used in the transfer
        # experiment (testing will be done on another dataset)
        emb_test, y_test, g_test = emb_train, y_train, g_train
    else:
        _, emb_train, y_train, g_train, cw, gw = dataset.get_split('train')
        _, emb_dev, y_dev, g_dev, _, _ = dataset.get_split('dev')
        _, emb_test, y_test, g_test, _, _ = dataset.get_split('test')
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_train)
        g_dev, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_dev)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

    train_model(emb_train, y_train, g_train, emb_dev, y_dev, g_dev, emb_test, y_test, g_test,
                checkpoint_file, params_file, clf_parameters, wrapper_parameters,
                cw, gw, max_epochs, dataset.multi_label)


def run(config):
    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    model_names = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    # predictions are saved for later, make sure the directory exists
    if not os.path.isdir(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])

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
                training_wrapper(config['checkpoint_dir'], dataset_setup['name'], model, pool, batch_size,
                                 config['embedding_dir'], config['classifier'], config['wrapper'],
                                 config['max_epochs'], lm_emb_size, local_dir=dataset_setup['local_dir'])


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('train_cbm.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_cbm.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
