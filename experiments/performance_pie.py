import os
import sys
import json
import pickle
import pandas as pd
import yaml
import scipy

import functools
import itertools
import getopt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from embedding import BertHuggingface

import copy
from salsa.SaLSA import SaLSA
from pie import TorchPipelineForEmbeddings

# local imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader
import models
import plotting
import utils


optimizer_lookup = {'Salsa': SaLSA, 'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW, 'Adamax': torch.optim.Adamax,
                    'Adadelta': torch.optim.Adadelta, 'Adagrad': torch.optim.Adagrad, 'SparseAdam': torch.optim.SparseAdam, 'ASGD': torch.optim.ASGD,
                    'SGD': torch.optim.SGD, 'LBFGS': torch.optim.LBFGS, 'RAdam': torch.optim.RAdam, 'NAdam': torch.optim.NAdam, 'Rprop': torch.optim.Rprop}
criterion_lookup = {'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss, 'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss, 'L1Loss': torch.nn.L1Loss, 'CrossEntropyLoss': torch.nn.CrossEntropyLoss, 'GaussianNLLLoss': torch.nn.GaussianNLLLoss,
              'MSELoss': torch.nn.MSELoss, 'CTCLoss': torch.nn.CTCLoss, 'NLLLoss': torch.nn.NLLLoss, 'PoissonNLLLoss': torch.nn.PoissonNLLLoss, 'KLDivLoss': torch.nn.KLDivLoss, 'BCELoss': torch.nn.BCELoss, 'MarginRankingLoss': torch.nn.MarginRankingLoss,
              'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss, 'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss, 'HuberLoss': torch.nn.HuberLoss, 'SmoothL1Loss': torch.nn.SmoothL1Loss, 'SoftMarginLoss': torch.nn.SoftMarginLoss,
              'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss, 'MultiMarginLoss': torch.nn.MultiMarginLoss, 'TripletMarginLoss': torch.nn.TripletMarginLoss, 'TripletMarginWithDistanceLoss': torch.nn.TripletMarginWithDistanceLoss}
clf_head_lookup = {'MLP2': models.MLP2Layer, 'linear': models.LinearClassifier, 'MLP3': models.MLP3Layer}


def create_pred_savefile_name(base_dir):
    now = datetime.now()
    file_suffix = now.strftime("%Y%m%d_%H%M%S.pickle")
    file_name = base_dir + file_suffix

    return file_name


def train_eval_one_split(emb_train: np.ndarray, y_train: np.ndarray, emb_val: np.ndarray, y_val: np.ndarray,
                         emb_test: np.ndarray, y_test: np.ndarray, g_test: np.ndarray, emb_def_attr: np.ndarray,
                         g_def: np.ndarray, group_match_lookup: dict, groups_test: list, groups_pie: list,
                         attr_lbl: list, clf_class: torch.nn.Module, cur_clf_params: dict, cur_wrapper_params: dict,
                         class_weights: np.ndarray, epochs: int, multi_label: bool) -> (float, float, float):
    clf_params = copy.deepcopy(cur_clf_params)
    n_features = emb_train.shape[1]
    clf_params['input_size'] = n_features
    if 'hidden_size_factor' in cur_clf_params.keys():
        assert (0 < clf_params['hidden_size_factor'] <= 1)
        if clf_class == models.MLP3Layer:
            # got 2 hidden layers
            clf_params['hidden_size1'] = int(n_features * clf_params['hidden_size_factor'])
            clf_params['hidden_size2'] = int(clf_params['hidden_size1'] * clf_params['hidden_size_factor'])
        else:
            clf_params['hidden_size'] = int(n_features * clf_params['hidden_size_factor'])
        clf_params.pop('hidden_size_factor', None)

    if not multi_label and y_train.ndim > 1:
        y_train = np.squeeze(y_train)
        y_val = np.squeeze(y_val)

    clf = clf_class(**clf_params)
    #wrapper = models.ClfWrapper(clf, **cur_wrapper_params, class_weights=class_weights)
    #epochs = wrapper.fit_early_stopping(emb_train, y_train, emb_val, y_val, max_epochs=epochs, delta=0.001, patience=10)
    #pred = wrapper.predict(emb_test)
    pipeline = TorchPipelineForEmbeddings(clf, **cur_wrapper_params, class_weights=class_weights)
    epochs = pipeline.fit_early_stopping(emb_protec=emb_def_attr, y_protec=g_def, emb_train=emb_train, y_train=y_train,
                                         emb_val=emb_val, y_val=y_val, attr_lbl=attr_lbl, group_lbl=groups_pie,
                                         max_epochs=epochs, delta=0.01, patience=10,)
    pred, concepts = pipeline.predict(emb_test, return_concepts=True, verbose=True)

    if multi_label:
        y_pred = (pred > 0.5).astype('int')
    else:
        y_pred = np.argmax(pred, axis=1)

    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    print("F1 score: %.2f, Precision: %.2f, Recall: %.2f" % (f1, prec, rec))

    # compute Pearson correlation of matching PIE concepts with the test groups
    corrs = []
    pvalues = []
    pie_matches = []
    for tid, group in enumerate(groups_test):
        matches = group_match_lookup[group]
        for match in matches:
            pid = groups_pie.index(match)
            pie_matches.append(pipeline.group_lbl[pid])  # format is "attr:group"
            r, p = scipy.stats.pearsonr(concepts[:, pid], g_test[:, tid])
            corrs.append(r)
            pvalues.append(p)

    clf.to_cpu()
    del clf

    return f1, prec, rec, corrs, pvalues, pie_matches, pred, concepts, epochs


def eval_cv(dataset: data_loader.CustomDataset, emb_def_attr: np.ndarray, g_def: np.ndarray,
            group_match_lookup: dict, groups_test: list, groups_pie: list, attr_lbl: list, clf_class: torch.nn.Module,
            cur_clf_params: dict, cur_wrapper_params: dict, max_epochs: int) -> (float, float, float):
    f1s = []
    precisions = []
    recalls = []
    rvalues = []
    pvalues = []
    all_predictions = []
    all_concepts = []
    epochs = []
    pie_matches = []
    for fold_id in range(dataset.n_folds):
        data_dict = dataset.get_cv_split(fold_id)
        X_train, emb_train, y_train, g_train, cw, gw = data_dict['train']
        X_test, emb_test, y_test, g_test, _, _ = data_dict['test']

        emb_train, emb_val, y_train, y_val = train_test_split(emb_train, y_train, test_size=0.1)
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, groups_test, g_train)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, groups_test, g_test)

        cur_f1, cur_prec, cur_rec, cur_corrs, cur_pval, pie_matches, \
            predictions, concepts, ep = train_eval_one_split(emb_train, y_train, emb_val, y_val, emb_test, y_test,
                                                             g_test, emb_def_attr, g_def, group_match_lookup,
                                                             groups_test, groups_pie, attr_lbl, clf_class,
                                                             cur_clf_params, cur_wrapper_params, cw, max_epochs,
                                                             dataset.multi_label)
        f1s.append(cur_f1)
        precisions.append(cur_prec)
        recalls.append(cur_rec)
        rvalues.append(cur_corrs)
        pvalues.append(cur_pval)
        all_predictions.append(predictions)
        all_concepts.append(concepts)
        epochs.append(ep)

    corrs = np.vstack(rvalues)
    ps = np.vstack(pvalues)

    return np.mean(f1s), np.mean(precisions), np.mean(recalls), np.mean(corrs, axis=0), np.mean(ps, axis=0), \
        pie_matches, np.vstack(all_predictions), np.vstack(all_concepts), epochs


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


def eval_all_clf_choices(results: pd.DataFrame, results_concepts: pd.DataFrame, defining_term_lookup: dict,
                         group_match_lookup: dict, dataset_name: str, model_name: str, pooling: str,
                         batch_size: int, emb_dir: str, clf_param_dict: dict, pred_dir: str, max_epochs: int,
                         local_dir: str = None) -> pd.DataFrame:
    """
    For one dataset and Backbone (model name + pooling) run the evaluation for all clf architecture and parameter
    choices.
    """

    # select sets of defining terms that are relevant for this dataset; then create "dataset" of defining terms and
    # groups labels
    cur_def_terms = utils.select_def_terms_for_dataset(defining_term_lookup, group_match_lookup)
    sel_groups = list(group_match_lookup.keys())
    attr_lbl = list(cur_def_terms.keys())

    print(dataset_name)
    print(sel_groups)
    print(attr_lbl)
    print(cur_def_terms)

    defining_terms, g_def, n_protected_concepts, groups_pie = utils.get_multi_attr_def_terms_labels(cur_def_terms)

    #groups_pie = []
    #for attr, terms_per_group in defining_term_lookup.items():
    #    groups_pie += list(terms_per_group.keys())

    model_type, model_architecture = utils.get_model_type_architecture(model_name)
    dataset, emb_def_attr = utils.get_dataset_with_embeddings(emb_dir, dataset_name, model_name, pooling, batch_size,
                                                              local_dir, defining_term_dict=defining_terms)
    print("run experiment for dataset %s, multi_label=%i" % (dataset_name, dataset.multi_label))

    # we apply CV for (small) datasets that do not provide a train-test split
    use_cv = (len(dataset.splits) == 1)

    # create list with classifier/ parameter configurations
    # add input and output size (matching the embedding size to clf parameters)
    clf_parameters = copy.deepcopy(clf_param_dict)
    clf_parameters['wrapper']['n_concepts_protec'] = n_protected_concepts
    n_concepts = clf_parameters['wrapper']['n_concepts_protec'] + clf_parameters['wrapper']['n_concepts_unsup']

    for key in clf_param_dict:
        if key != 'wrapper':
            print("set input and output dim for clf: " + key)
            clf_parameters[key]['input_size'] = n_concepts
            clf_parameters[key]['output_size'] = dataset.n_classes
    clf_parameters['wrapper']['optimizer'] = [optimizer_lookup[optim] for optim in
                                              clf_param_dict['wrapper']['optimizer']]
    if dataset.multi_label:
        clf_parameters['wrapper']['criterion'] = [criterion_lookup[loss] for loss in
                                                  clf_param_dict['wrapper']['criterion']['multi-label']]
    else:
        clf_parameters['wrapper']['criterion'] = [criterion_lookup[loss] for loss in
                                                  clf_param_dict['wrapper']['criterion']['single-label']]

    print("clf parameters:")
    print(clf_parameters)

    # get all possible set of parameters for the classifiers
    clf_parameter_sets = {}
    for key, param_dict in clf_parameters.items():
        clf_parameter_sets[key] = get_parameter_sets(param_dict)
    print("clf parameter sets:")
    print(clf_parameter_sets)

    if not use_cv:
        X_train, emb_train, y_train, g_train, cw, gw = dataset.get_split('train')
        X_dev, emb_dev, y_dev, g_dev, _, _ = dataset.get_split('dev')
        X_test, emb_test, y_test, g_test, _, _ = dataset.get_split('test')
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_train)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

    classifier_choices = [key for key in clf_parameters.keys() if key != 'wrapper']
    for clf in classifier_choices:
        for clf_params in clf_parameter_sets[clf]:
            for wrapper_params in clf_parameter_sets['wrapper']:
                try:  # salsa might fail
                    if use_cv:
                        f1, prec, rec, corr, pval, sel_groups_pie, \
                            predictions, concepts, ep = eval_cv(dataset, emb_def_attr, g_def, group_match_lookup,
                                                                sel_groups, groups_pie, attr_lbl, clf_head_lookup[clf],
                                                                clf_params, wrapper_params, max_epochs)
                    else:
                        f1, prec, rec, corr, pval, sel_groups_pie, \
                            predictions, concepts, ep = train_eval_one_split(emb_train, y_train, emb_dev, y_dev,
                                                                             emb_test, y_test, g_test, emb_def_attr,
                                                                             g_def, group_match_lookup, sel_groups,
                                                                             groups_pie, attr_lbl,
                                                                             clf_head_lookup[clf], clf_params,
                                                                             wrapper_params, cw, max_epochs,
                                                                             dataset.multi_label)

                    # save predictions (for CV concatenate all predictions):
                    save_dict = {'predictions': predictions}
                    file_name = create_pred_savefile_name(pred_dir)
                    with open(file_name, "wb") as handle:
                        pickle.dump(save_dict, handle)
                except RuntimeError:
                    print("learning failed for %s on %s" % (model_name, dataset_name))
                    f1 = 0
                    prec = 0
                    rec = 0
                    ep = 0
                    file_name = 'na'
                    sel_groups_pie = []  # no concept results will be written to the csv

                hidden_size = -1
                if 'hidden_size_factor' in clf_params.keys():
                    hidden_size = clf_params['hidden_size_factor']
                optim = list(optimizer_lookup.keys())[list(optimizer_lookup.values()).index(wrapper_params['optimizer'])]
                loss_fct = list(criterion_lookup.keys())[list(criterion_lookup.values()).index(wrapper_params['criterion'])]
                emb_dim = dataset.data_preprocessed[dataset.splits[0]].shape[1]

                # performance results (only one row per dataset and clf/wrapper params
                results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture, 'baseline',
                                                   pooling, clf, hidden_size, emb_dim, optim, wrapper_params['lr'],
                                                   loss_fct, f1, prec, rec, ep, file_name]

                # concept results (one row for each protected group)
                # if training failed sel_groups_pie will be empty and no results will be written
                for i, (group_test, group_pie) in enumerate(zip(sel_groups, sel_groups_pie)):
                    results_concepts.loc[len(results_concepts.index)] = [dataset_name, model_name, model_type,
                                                                         model_architecture, 'pie', pooling, clf,
                                                                         hidden_size, emb_dim, optim,
                                                                         wrapper_params['lr'], loss_fct,
                                                                         group_pie, group_test, corr[i], pval[i], ep,
                                                                         file_name]

    return results


def run(config):

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    model_names = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    # predictions are saved for later, make sure the directory exists
    if not os.path.isdir(config["pred_dir"]):
        os.makedirs(config["pred_dir"])

    # read config with defining terms (for PIE) and label matches with the datasets
    with open(config['def_term_file'], 'r') as ff:
        def_term_config = yaml.safe_load(ff)
    defining_term_lookup = def_term_config['defining_terms']
    group_match_lookup = def_term_config['group_matching']

    # prepare directory where results will be saved
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])

    # results for performance
    result_keys = ["dataset", "model", "model type", "architecture", "method", "pooling", "classifier",
                   "clf hidden size factor", "emb size", "optimizer",
                   "lr", "loss", "F1", "Precision", "Recall", "Epochs", "Predictions"]
    results_path = config['results_dir'] + 'pie_performance_results.csv'
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame({key: [] for key in result_keys})

    # results for concepts
    result_concept_keys = ["dataset", "model", "model type", "architecture", "method", "pooling", "classifier",
                           "clf hidden size factor", "emb size", "optimizer", "lr", "loss",
                           "group (pie)", "group (test)", "Pearson R", "pvalue", "Epochs", "concepts"]
    results_concept_path = config['results_dir'] + 'pie_concept_results.csv'
    if os.path.isfile(results_concept_path):
        results_concept = pd.read_csv(results_concept_path)
    else:
        results_concept = pd.DataFrame({key: [] for key in result_concept_keys})

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
                    cur_params = [dataset_setup['name'], model]
                else:
                    cur_params = [dataset_setup['name'], model, pool]
                cur_params_per_key = dict(zip(result_keys[:len(cur_params)], cur_params))
                result_filter = functools.reduce(lambda a, b: a & b,
                                                 [(results[key] == val) for key, val in cur_params_per_key.items()])

                if results.loc[result_filter].empty:
                    results = eval_all_clf_choices(results, results_concept, defining_term_lookup,
                                                   group_match_lookup[dataset_setup['name']],
                                                   dataset_setup['name'], model, pool, batch_size,
                                                   config['embedding_dir'], config['classifier'], config['pred_dir'],
                                                   config['max_epochs'], local_dir=dataset_setup['local_dir'])
                    print("save results for setup: %s, %s, %s" % (dataset_setup['name'], model, pool))
                    results.to_csv(results_path, index=False)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('performance_baseline.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('performance_baseline.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run(config)


if __name__ == "__main__":
    main(sys.argv[1:])
