import os
import sys
import json
import pickle
import pandas as pd
import scipy

import functools
import itertools
import yaml
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
import utils

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


def get_standardized_label(label_match: dict, label: str):
    for attr, label_dict in label_match.items():
        for standard_lbl, labels in label_dict.items():
            if label in labels:
                return standard_lbl


def evaluate_cbm(cbmWrapper: models.CBMWrapper, dataset_train: str, dataset_test: str, model_name: str,
                 pooling: str, batch_size: int, emb_dir: str, train_groups: list, label_matches: dict,
                 local_dir: str = None):
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset_test, model_name, pooling, batch_size, local_dir)
    print("evaluate CBM trained on %s on %s" % (dataset_train, dataset_test))

    if len(dataset.splits) == 1:
        _, emb_test, y_test, g_test, _, _ = dataset.get_split(dataset.splits[0])
        #g_test, groups, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

    else:
        _, emb_test, y_test, g_test, _, _ = dataset.get_split('test')
        #g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

    # standardize dataset-specific group label and find shared label indices
    eval_groups_s = [get_standardized_label(label_matches, lbl) for lbl in dataset.group_names]
    train_groups_s = [get_standardized_label(label_matches, lbl) for lbl in train_groups]
    shared_lbl = [lbl for lbl in eval_groups_s if lbl in train_groups_s and lbl is not None]
    if len(shared_lbl) == 0:
        print("datasets %s and %s do not share any labels" % (dataset_train, dataset_test))
        return [], [], [], [], []

    if None in shared_lbl or 'None' in shared_lbl:
        print(shared_lbl)
        exit(0)

    eval_ids = [eval_groups_s.index(lbl) for lbl in shared_lbl]
    train_ids = [train_groups_s.index(lbl) for lbl in shared_lbl]

    # apply CBM to eval data
    pred, concepts = cbmWrapper.predict(emb_test)

    # evaluate concept alignment
    rs = []
    ps = []
    aucs = []
    for i in range(len(eval_ids)):
        cbm_idx = train_ids[i]
        eval_idx = eval_ids[i]
        r, p = scipy.stats.pearsonr(g_test[:, eval_idx], concepts[:, cbm_idx])
        precision, recall, thresh = precision_recall_curve(g_test[:, eval_idx], concepts[:, cbm_idx])
        rs.append(r)
        ps.append(p)
        aucs.append(auc(recall, precision))

    return rs, ps, aucs, [dataset.group_names[idx] for idx in eval_ids], [train_groups[idx] for idx in train_ids]


def run(config):
    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    model_names = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    # make sure the directory for results exists
    results_path = config['results_path']
    results_dir = results_path.replace(results_path.split('/')[-1], '')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    with open(config['label_match_config'], 'r') as ff:
        label_match_config = yaml.safe_load(ff)

    # results for performance
    result_keys = ["dataset (train)", "dataset (test)", "model", "model type", "architecture",
                   "method", "pooling", "classifier", "clf hidden size factor",
                   "emb size", "protected concepts", "other concepts",
                   "lambda", "optimizer", "lr", "loss", "group (train)", "group (test)",
                   "Pearson R", "pvalue", "PR-AUC"]
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
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
        model_type, model_architecture = utils.get_model_type_architecture(model)
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        if model in openai_models:
            pooling_choices = ['']

        for pool in pooling_choices:
            for dataset_setup in config['datasets']:
                train_dataset = dataset_setup['name']
                checkpoint_file, params_file = get_cbm_savefile(config['checkpoint_dir'], train_dataset, model, pool,
                                                                file_suffix=None)
                if not os.path.isfile(checkpoint_file) and os.path.isfile(params_file):
                    print("checkpoint and/ or parameter file not found: %s, %s" % (checkpoint_file, params_file))
                    print("skip %s as training dataset" % train_dataset)
                    continue

                # load the pre-trained CBM
                with open(params_file, "rb") as handle:
                    model_params = pickle.load(handle)
                hidden_size = model_params['clf']['hidden_size'] if 'hidden_size' in model_params['clf'].keys() else -1

                cbm = models.CBM(**model_params['clf'])
                cbm.load_state_dict(torch.load(checkpoint_file, weights_only=True))
                cbmWrapper = models.CBMWrapper(cbm, **model_params['wrapper'])

                dataset = data_loader.get_dataset(train_dataset, dataset_setup['local_dir'])

                for eval_setup in config['datasets']:
                    eval_dataset = eval_setup['name']

                    # only cross-dataset transfer
                    # (except twitterAAE which is not in the performance experiment)
                    if eval_dataset == train_dataset and not eval_dataset == 'twitterAAE':
                        continue

                    # check if results exist
                    if pool == '':
                        cur_params_per_key = {'dataset (train)': train_dataset, 'dataset (test)': eval_dataset,
                                              'model': model}
                    else:
                        cur_params_per_key = {'dataset (train)': train_dataset, 'dataset (test)': eval_dataset,
                                              'model': model, 'pooling': pool}
                    result_filter = functools.reduce(lambda a, b: a & b, [(results[key] == val)
                                                                          for key, val in cur_params_per_key.items()])
                    if results.loc[result_filter].empty:
                        # run eval
                        (rs, ps, aucs,
                         eval_groups, train_groups) = evaluate_cbm(cbmWrapper, train_dataset, eval_dataset, model,
                                                                   pool, batch_size, config['embedding_dir'],
                                                                   dataset.group_names, label_match_config,
                                                                   eval_setup['local_dir'])
                        # save results
                        for i in range(len(rs)):
                            results.loc[len(results.index)] = [train_dataset, eval_dataset, model, model_type,
                                                               model_architecture, 'cbm', pool,
                                                               config['wrapper']['clf'], hidden_size, lm_emb_size,
                                                               model_params['clf']['n_concepts_protec'],
                                                               model_params['clf']['n_concepts_unsup'],
                                                               model_params['wrapper']['lambda_concept'],
                                                               model_params['wrapper']['optimizer'],
                                                               model_params['wrapper']['lr'],
                                                               model_params['wrapper']['criterion'], train_groups[i],
                                                               eval_groups[i], rs[i], ps[i], aucs[i]]

                        print("save results")
                        results.to_csv(results_path, index=False)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('transfer_cbm.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('transfer_cbm.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
