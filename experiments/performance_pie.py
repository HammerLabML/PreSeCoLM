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
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
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
    
    # set clf input size
    n_concepts = cur_wrapper_params['n_concepts_protec'] + cur_wrapper_params['n_concepts_unsup']
    clf_params['input_size'] = n_concepts
    
    if 'hidden_size_factor' in cur_clf_params.keys():
        assert (0 < clf_params['hidden_size_factor'] <= 1)
        if clf_class == models.MLP3Layer:
            # got 2 hidden layers
            clf_params['hidden_size1'] = int(n_concepts * clf_params['hidden_size_factor'])
            clf_params['hidden_size2'] = int(clf_params['hidden_size1'] * clf_params['hidden_size_factor'])
        else:
            clf_params['hidden_size'] = int(n_concepts * clf_params['hidden_size_factor'])
        clf_params.pop('hidden_size_factor', None)

    #print(clf_params)
    #print(cur_wrapper_params)

    if not multi_label and y_train.ndim > 1:
        y_train = np.squeeze(y_train)
        y_val = np.squeeze(y_val)

    clf = clf_class(**clf_params)
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

    # groups_pie is a list of lists -> create one list for matches
    groups_pie = list(itertools.chain(*groups_pie))
    print(groups_test)
    print(groups_pie)

    # extract the group names form the pipeline (format attr:group)
    group_ids_pipeline = {} #= pipeline.group_lbl #[]
    for group in groups_pie:
        for i, group2 in enumerate(pipeline.group_lbl):
            if group2.split(':')[1] == group:
                group_ids_pipeline[group] = i
    print(group_ids_pipeline)

    # compute Pearson correlation of matching PIE concepts with the test groups
    corrs = []
    pvalues = []
    aucs = []
    pie_matches = []
    groups_gt = []
    for tid, group in enumerate(groups_test):
        matches = group_match_lookup[group]
        for match in matches:
            pid = group_ids_pipeline[match]
            pie_matches.append(pipeline.group_lbl[pid])
            groups_gt.append(group)
            r, p = scipy.stats.pearsonr(concepts[:, pid], g_test[:, tid])
            precision, recall, thresh = precision_recall_curve(g_test[:, tid], concepts[:, pid])
            corrs.append(r)
            pvalues.append(p)
            aucs.append(auc(recall, precision))

    clf.to_cpu()
    del clf

    print(groups_gt)
    print("PR-AUC:", aucs)


    # return only the protected concepts
    n_protected_concepts = len(pipeline.group_lbl)
    concepts_ret = concepts[:, :n_protected_concepts]


    return f1, prec, rec, corrs, pvalues, aucs, pipeline.group_lbl, pie_matches, groups_gt, pred, concepts_ret, epochs


def eval_cv(dataset: data_loader.CustomDataset, emb_def_attr: np.ndarray, g_def: np.ndarray,
            group_match_lookup: dict, groups_test: list, groups_pie: list, attr_lbl: list, clf_class: torch.nn.Module,
            cur_clf_params: dict, cur_wrapper_params: dict, max_epochs: int) -> (float, float, float):
    f1s = []
    precisions = []
    recalls = []
    rvalues = []
    pvalues = []
    aucs = []
    pie_label = []
    all_predictions = []
    all_concepts = []
    epochs = []
    pie_matches = []
    groups_gt = []
    for fold_id in range(dataset.n_folds):
        data_dict = dataset.get_cv_split(fold_id)
        X_train, emb_train, y_train, g_train, cw, gw = data_dict['train']
        X_test, emb_test, y_test, g_test, _, _ = data_dict['test']

        emb_train, emb_val, y_train, y_val = train_test_split(emb_train, y_train, test_size=0.1)
        g_train, groups, _ = utils.filter_group_labels(dataset.group_names, groups_test, g_train)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, groups_test, g_test)

        (cur_f1, cur_prec, cur_rec, cur_corrs, cur_pval, cur_auc, pie_label,
         pie_matches, groups_gt, predictions, concepts, ep) = train_eval_one_split(emb_train, y_train, emb_val, y_val, emb_test, y_test,
                                                             g_test, emb_def_attr, g_def, group_match_lookup,
                                                             groups_test, groups_pie, attr_lbl, clf_class,
                                                             cur_clf_params, cur_wrapper_params, cw, max_epochs,
                                                             dataset.multi_label)
        f1s.append(cur_f1)
        precisions.append(cur_prec)
        recalls.append(cur_rec)
        rvalues.append(cur_corrs)
        pvalues.append(cur_pval)
        aucs.append(cur_auc)
        all_predictions.append(predictions)
        all_concepts.append(concepts)
        epochs.append(ep)

    corrs = np.vstack(rvalues)
    ps = np.vstack(pvalues)

    return np.mean(f1s), np.mean(precisions), np.mean(recalls), np.mean(corrs, axis=0), np.mean(ps, axis=0), \
        np.mean(aucs, axis=0), pie_label, pie_matches, groups_gt, np.vstack(all_predictions), np.vstack(all_concepts), epochs


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
                         emb_dim: int, local_dir: str = None) -> pd.DataFrame:
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
    #n_concepts = clf_parameters['wrapper']['n_concepts_protec'] + clf_parameters['wrapper']['n_concepts_unsup'][0]

    for key in clf_param_dict:
        if key != 'wrapper':
            print("set input and output dim for clf: " + key)
            #clf_parameters[key]['input_size'] = n_concepts
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
        n_samples_train = len(X_train)
    else:
        data_dict = dataset.get_cv_split(0)
        n_samples_train = len(data_dict['train'][0])

    classifier_choices = [key for key in clf_parameters.keys() if key != 'wrapper']
    for clf in classifier_choices:
        for clf_params in clf_parameter_sets[clf]:
            for wrapper_params in clf_parameter_sets['wrapper']:
                if wrapper_params['n_concepts_unsup'] == -1:
                    wrapper_params['n_concepts_unsup'] = emb_dim - wrapper_params['n_concepts_protec']
                    print(wrapper_params['n_concepts_unsup'])
                n_concepts = wrapper_params['n_concepts_protec'] + wrapper_params['n_concepts_unsup']

                if n_concepts > emb_def_attr.shape[1]:
                    print("skip parameter set with n_concepts_unsup=%i, bc the number of concepts exceeds the embedding size" % wrapper_params['n_concepts_unsup'])
                    continue
                if n_concepts > n_samples_train:
                    print("skip parameter set with n_concepts_unsup=%i, bc the number of concepts exceeds the number of training samples" % wrapper_params['n_concepts_unsup'])
                    continue

                try:  # salsa might fail
                    if use_cv:
                        f1, prec, rec, corr, pval, aucs, pie_label, sel_groups_pie, groups_gt, \
                            predictions, concepts, ep = eval_cv(dataset, emb_def_attr, g_def, group_match_lookup,
                                                                sel_groups, groups_pie, attr_lbl, clf_head_lookup[clf],
                                                                clf_params, wrapper_params, max_epochs)
                    else:
                        f1, prec, rec, corr, pval, aucs, pie_label, sel_groups_pie, groups_gt, \
                            predictions, concepts, ep = train_eval_one_split(emb_train, y_train, emb_dev, y_dev,
                                                                             emb_test, y_test, g_test, emb_def_attr,
                                                                             g_def, group_match_lookup, sel_groups,
                                                                             groups_pie, attr_lbl,
                                                                             clf_head_lookup[clf], clf_params,
                                                                             wrapper_params, cw, max_epochs,
                                                                             dataset.multi_label)

                    # save predictions (for CV concatenate all predictions):
                    save_dict = {'predictions': predictions, 'concepts': concepts, 'groups_pie': pie_label}
                    file_name = create_pred_savefile_name(pred_dir)
                    with open(file_name, "wb") as handle:
                        pickle.dump(save_dict, handle)
                except ValueError as error:
                    print("learning failed for %s on %s" % (model_name, dataset_name))
                    print(error)
                    f1 = 0
                    prec = 0
                    rec = 0
                    ep = 0
                    file_name = 'na'
                    sel_groups_pie = []  # no concept results will be written to the csv
                    groups_gt = []
                except RuntimeError as error:
                    print("learning failed for %s on %s" % (model_name, dataset_name))
                    print(error)
                    f1 = 0
                    prec = 0
                    rec = 0
                    ep = 0
                    file_name = 'na'
                    sel_groups_pie = []  # no concept results will be written to the csv
                    groups_gt = []

                hidden_size = -1
                if 'hidden_size_factor' in clf_params.keys():
                    hidden_size = clf_params['hidden_size_factor']
                optim = list(optimizer_lookup.keys())[list(optimizer_lookup.values()).index(wrapper_params['optimizer'])]
                loss_fct = list(criterion_lookup.keys())[list(criterion_lookup.values()).index(wrapper_params['criterion'])]
                emb_dim = dataset.data_preprocessed[dataset.splits[0]].shape[1]

                # performance results (only one row per dataset and clf/wrapper params
                results.loc[len(results.index)] = [dataset_name, model_name, model_type, model_architecture, 'pie',
                                                   pooling, clf, hidden_size, emb_dim, n_protected_concepts,
                                                   wrapper_params['n_concepts_unsup'], wrapper_params['method_protec'],
                                                   wrapper_params['method_unsup'], wrapper_params['remove_protected_features'], optim, wrapper_params['lr'],
                                                   loss_fct, f1, prec, rec, ep, file_name]

                # concept results (one row for each protected group)
                # if training failed sel_groups_pie will be empty and no results will be written
                for i, (group_test, group_pie) in enumerate(zip(groups_gt, sel_groups_pie)):
                    results_concepts.loc[len(results_concepts.index)] = [dataset_name, model_name, model_type,
                                                                         model_architecture, 'pie', pooling, clf,
                                                                         hidden_size, emb_dim, n_protected_concepts,
                                                                         wrapper_params['n_concepts_unsup'],
                                                                         wrapper_params['method_protec'],
                                                                         wrapper_params['method_unsup'],
                                                                         wrapper_params['remove_protected_features'],
                                                                         optim, wrapper_params['lr'], loss_fct,
                                                                         group_pie, group_test, corr[i], pval[i],
                                                                         aucs[i], ep, file_name]

    return results


def run(config):

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    model_names = openai_models + huggingface_models

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
                   "clf hidden size factor", "emb size", "protected concepts", "other concepts",
                   "method protected", "method unsupervised", "remove protected", "optimizer",
                   "lr", "loss", "F1", "Precision", "Recall", "Epochs", "Predictions"]
    results_path = config['results_dir'] + 'pie_performance_results.csv'
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame({key: [] for key in result_keys})

    # results for concepts
    result_concept_keys = ["dataset", "model", "model type", "architecture", "method", "pooling", "classifier",
                           "clf hidden size factor", "emb size", "protected concepts", "other concepts",
                           "method protected", "method unsupervised", "remove protected",
                           "optimizer", "lr", "loss", "group (pie)", "group (test)",
                           "Pearson R", "pvalue", "PR-AUC", "Epochs", "concepts"]

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
                    results = eval_all_clf_choices(results, results_concept, defining_term_lookup,
                                                   group_match_lookup[dataset_setup['name']],
                                                   dataset_setup['name'], model, pool, batch_size,
                                                   config['embedding_dir'], config['classifier'], config['pred_dir'],
                                                   config['max_epochs'], lm_emb_size,
                                                   local_dir=dataset_setup['local_dir'])
                    print("save results for setup: %s, %s, %s" % (dataset_setup['name'], model, pool))
                    results.to_csv(results_path, index=False)
                    results_concept.to_csv(results_concept_path, index=False)


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
