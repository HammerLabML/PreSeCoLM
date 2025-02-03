import os
import sys
import json
import pickle
from tqdm import tqdm
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import functools
import getopt
import numpy as np
import math
import random
import scipy
from sklearn.metrics import f1_score, precision_score, recall_score

import torch

from pie_data import get_dataset, label2onehot
from models import CBM, CBMWrapper
import plotting
import utils


def get_cbm_savefile(cbm_dir, dataset, model, pooling, file_suffix=None):
    model = model.replace('/','_')
    if file_suffix is not None:
        checkpoint_file = cbm_dir + ("%s_%s_%s" % (dataset, model, pooling)) + file_suffix
        params_file = cbm_dir + ("%s_%s" % (dataset, model)) + file_suffix
    else:
        checkpoint_file = cbm_dir + ("%s_%s_%s" % (dataset, model, pooling))
        params_file = cbm_dir + ("%s_%s" % (dataset, model))
    return checkpoint_file, params_file

def train_cbms(dataset, model, pooling, batch_size, emb_dir, cbm_dir, plot_dir, local_dir=None, sel_groups=None, file_suffix=None, lambda_concept=0.5, use_concept_weights=False, lr=1e-4, epochs=80):
    # check if checkpoint for this CBM exists already
    checkpoint_file, params_file = get_cbm_savefile(cbm_dir, dataset, model, pooling, file_suffix)
    if os.path.isfile(checkpoint_file) and os.path.isfile(params_file):
        print("cbm checkpoint and parameter file for %s, %s, %s, %s already exists" % (dataset, model, pooling, file_suffix))
        return

    # get data and embeddings
    X_train, emb_train, y_train, g_train, X_test, emb_test, y_test, g_test, groups, _, class_weights = utils.get_dataset_and_embeddings(emb_dir, dataset, model, pooling, batch_size, local_dir, sel_groups=sel_groups)
    n_classes, multi_label = utils.get_number_of_classes(y_test)
    
    if sel_groups is not None:
        assert sel_groups == groups, ("expected %s, instead got %s " % (sel_groups, groups))

    # data_loader with binary group labels provide only 1D labels for two groups; adjust label accordingly
    # in this case, group labels will be learned as single label and are converted to multi-label for evaluation
    if len(groups) == 2 and (g_test.ndim == 1 or g_test.shape[1] == 1):
        groups = [groups[0] + '/' + groups[1]]

    # determine criterion and prepare class labels for torch model; for multi-labels class weights are applied
    print("fit CBM for %s, %s, %s (%s)" % (dataset, model, pooling, file_suffix))
    print("with groups ", groups)
    if multi_label:
        print("multi label: use BCEWithLogitsLoss")
        criterion = torch.nn.BCEWithLogitsLoss
    else:
        print("single label: use CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss
        class_weights=None
        y_train = y_train.flatten().astype('int')
        y_test = y_test.flatten().astype('int')

    # some data_loader do not have a training split
    if len(g_train) == 0 or len(y_train) == 0:
        print("got no training data to fit CBM")
        return

    # compute concept weights, prepare concept labels
    if use_concept_weights:
        concept_weights = compute_class_weights(g_train, sel_groups)
    else:
        concept_weights = None
    c_train = label2onehot(g_train)
    c_test = label2onehot(g_test)

    # init model params
    model_params = {'input_size': emb_train.shape[1], 'output_size': n_classes, 'n_learned_concepts': c_train.shape[1], 'n_other_concepts': emb_train.shape[1]-c_train.shape[1], 'hidden_size': 300}
    print("train CBM with params:")
    print(model_params)
    
    cbm = CBM(**model_params)
    cbmWrapper = CBMWrapper(cbm, batch_size=32, class_weights=class_weights, criterion=criterion, 
                            concept_criterion=torch.nn.BCEWithLogitsLoss, lr=lr, lambda_concept=lambda_concept, concept_weights=concept_weights)

    # train and remember class/ concept scores for each epoch
    f1s_c = []
    f1s_p = []
    for e in tqdm(range(epochs)):
        cbmWrapper.fit(emb_train, y_train, c_train, epochs=1)
        pred, concepts = cbmWrapper.predict(emb_test)

        if multi_label:
            y_pred = (np.array(pred) >= 0.5).astype(int)
        else: 
            y_pred = np.argmax(pred, axis=1)
        c_pred = (np.array(concepts) >= 0.5).astype(int)
        
        f1_concept = f1_score(c_test, c_pred, average='macro')
        f1_pred = f1_score(y_test, y_pred, average='macro')
        f1s_c.append(f1_concept)
        f1s_p.append(f1_pred)

    print(f1s_p)
    learning_curve_dir = '%s/cbm_learning_curves/' % plot_dir
    if not os.path.isdir(learning_curve_dir):
        os.makedirs(learning_curve_dir)

    # plot learning curve for class and concept F1-score
    x = np.arange(len(f1s_c))
    fig, ax = plt.subplots()
    ax.plot(x,f1s_c,label='concept F1')
    ax.plot(x,f1s_p,label='pred F1')
    ax.legend()
    savefile = ('%s/%s_%s_%s_%s.png' % (learning_curve_dir, dataset, model.replace('/','_'), pooling, file_suffix))
    plt.savefig(savefile, bbox_inches='tight')

    # save model
    torch.save(cbm.state_dict(), checkpoint_file)
    with open(params_file, 'wb') as handle:
        pickle.dump(model_params, handle)


def evaluate_cbms(dataset_train, dataset_test, model, pooling, batch_size, emb_dir, cbm_dir, plot_dir, local_dir=None, sel_groups_train=None, sel_groups_test=None, file_suffix=None):
    # verify checkpoint exists
    checkpoint_file, params_file = get_cbm_savefile(cbm_dir, dataset_train, model, pooling, '')
    if not os.path.isfile(checkpoint_file):
        print("could not find CBM checkpoint for %s %s %s" % (dataset_train, model, pooling))
        print(checkpoint_file)
        return

    # load test dataset and get embeddings
    _, _, _, _, X_test, emb_test, y_test, g_test, groups_test, _, _ = utils.get_dataset_and_embeddings(emb_dir, dataset_test, model, pooling, batch_size, local_dir, sel_groups=sel_groups_test)
    n_classes, multi_label = utils.get_number_of_classes(y_test)

    # concepts encoded as onehot
    c_test = label2onehot(g_test)

    # with onehot encoding number of test groups and shape of test concepts should match
    # the number of training groups might be larger
    assert c_test.shape[1] <= len(sel_groups_train) and c_test.shape[1] == len(groups_test)

    # load model and parameters
    with open(params_file, "rb") as handle:
        model_params = pickle.load(handle)
        
    cbm = CBM(**model_params)
    with open(checkpoint_file, "rb") as handle:
        cbm.load_state_dict(torch.load(checkpoint_file, weights_only=True))
        
    # wrapper only needed for predict call, most params don't matter
    cbmWrapper = CBMWrapper(cbm, batch_size=32, class_weights=None, criterion=torch.nn.BCEWithLogitsLoss, 
                            concept_criterion=torch.nn.BCEWithLogitsLoss, lr=1e-4, lambda_concept=0.5, concept_weights=None)

    # get concept predictions
    _, concepts = cbmWrapper.predict(emb_test)
    c_pred = (np.array(concepts) >= 0).astype(int) # concepts are logits!

    # handle different dimensionality of test and predicted ocncepts
    if not utils.is1D(c_test) and utils.is1D(c_pred):
        print("convert 1d prediction to onehot")
        c_pred = label2onehot(c_pred, minv=int(np.min(c_test)), maxv=int(np.max(c_test)))
        concepts = np.hstack([-concepts, concepts])
        assert len(sel_groups_train) == 2
    if utils.is1D(c_test) and not utils.is1D(c_pred):
        print("convert 1d test labels to onehot")
        c_test = label2onehot(c_test)

    # evaluate the concept predictions
    # if test and train labels do not exactly match, this can be handled by utils.eval_with_label_match
    # otherwise just compute F1-score and Pearson correlation
    if len(sel_groups_train) != len(groups_test):
        f1, corr, groups_train_ordered = utils.eval_with_label_match(c_test, c_pred, concepts, sel_groups_test, sel_groups_train, average='macro')
    else:
        f1 = f1_score(c_test, c_pred, average='macro')
        groups_train_ordered = groups_test

        if not utils.is1D(concepts): # both > 1D
            corr = scipy.stats.pearsonr(concepts, np.asarray(c_test))
        else: # both 1D
            corr = scipy.stats.pearsonr(concepts.flatten(), c_test.flatten())

    # plot histograms for all test groups (1 or 0) against all predicted concepts, save plot
    model_name = model.replace('/','_')
    plot_dir = '%s/cbm_eval' % plot_dir
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if file_suffix is not None:
        plot_savefile = ('%s/%s_%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling, file_suffix))
    else:
        plot_savefile = ('%s/%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling))
    plotting.plot_feature_histogram(concepts, c_test, labels=groups_test, features=sel_groups_train, xlabel=dataset_train, ylabel=dataset_test, savefile=plot_savefile)

    return f1, corr, groups_train_ordered, groups_test

def run_cbm_training(config):
    # config with training setups (data_loader, selected groups...)
    with open(config["cbm_train_config"], 'r') as stream:
        training_setups = yaml.safe_load(stream)

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    if not os.path.isdir(config["cbm_dir"]):
        os.makedirs(config["cbm_dir"])

    # train the CBM for any combination of LM, pooling and dataset
    for model in models:
        pooling_choices = config["pooling"]
        batch_size = 1
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        if model in openai_models:
            pooling_choices = ['']
        for pool in pooling_choices:
            for setup in training_setups:
                # these data_loader do not have a classification labels, so a CBM cannot be trained
                if setup['dataset'] in ['twitterAAE', 'crows_pairs']:
                    continue
                train_cbms(setup['dataset'], model, pool, batch_size, config["embedding_dir"], config["cbm_dir"], plot_dir=config["plot_dir"],
                           local_dir=setup['local_dir'], sel_groups=setup['groups'], file_suffix=setup['suffix'])

def run_cbm_eval(config):
    # prepare directory where results will be saved
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])

    # config with evaluation setups (data_loader, protected attributes and groups...)
    with open(config['eval_config'], 'r') as stream:
        eval_setups_by_attr = yaml.safe_load(stream)
    with open(config['cbm_train_config'], 'r') as stream:
        train_setups = yaml.safe_load(stream)

    # language models
    openai_models = config["openai_models"]
    huggingface_models = config["huggingface_models"]
    models = huggingface_models + openai_models

    # dictionary with batch sizes for huggingface models
    with open(config["batch_size_lookup"], 'r') as f:
        batch_size_lookup = json.load(f)

    result_keys_cbm = ["protected_attr", "dataset (train)", "dataset (eval)", "embedder", "pooling", "group (test)", "concept (train)", "F1", "Pearson R", "Pearson p"]

    # read existing results or create new dataframe
    results_cbm_path = config['results_dir'] + config['cbm_results_file']
    if os.path.isfile(results_cbm_path):
        results_cbm = pd.read_csv(results_cbm_path)
    else:
        results_cbm = pd.DataFrame({key: [] for key in result_keys_cbm})

    # evaluate CBMs for the different protected attributes, data_loader, models...
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
                for train_setup in train_setups:
                    # these data_loader do not have a classification labels, so a CBM cannot be trained
                    if train_setup['dataset'] in ['twitterAAE', 'crows_pairs']:
                        continue
                    if model == 'text-embedding-3-large' and train_setup['dataset'] in ['jigsaw', 'twitterAAE']: # TODO remove later
                        continue

                    # check if train dataset relevant for the current attribute
                    dataset_in_test_case = False
                    for test_setup in eval_setups:
                        if test_setup['dataset'] == train_setup['dataset']:
                            dataset_in_test_case = True
                    if not dataset_in_test_case:
                        continue
                    
                    for eval_setup in eval_setups:
                        if model == 'text-embedding-3-large' and eval_setup['dataset'] in ['jigsaw', 'twitterAAE']: # TODO remove later
                            continue
                            
                        # skip 'same dataset' cases where no training data is available
                        if train_setup['dataset'] == eval_setup['dataset'] and eval_setup['dataset'] in ['twitterAAE', 'crows_pairs']:
                            continue  # no training data available for twitterAAE and crowspairs (TODO CV on test data)

                        # create filter to determine if this setup has already been evaluated
                        if pool == '':
                            cur_params = [attr, train_setup['dataset'], eval_setup['dataset'], model]
                        else:
                            cur_params = [attr, train_setup['dataset'], eval_setup['dataset'], model, pool]
                        cur_params_per_key = dict(zip(result_keys_cbm[:len(cur_params)], cur_params))
                        result_filter = functools.reduce(lambda a, b: a & b, [(results_cbm[key] == val) for key, val in cur_params_per_key.items()])

                        if results_cbm.loc[result_filter].empty:
                            # this combination of train and eval setup has not been evaluated yet
                            print("testing %s features on %s" % (train_setup['dataset'], eval_setup['dataset']))

                            # run eval
                            f1, corr, groups_train, groups_eval = evaluate_cbms(train_setup['dataset'], eval_setup['dataset'], model, pool, batch_size, config["embedding_dir"], config["cbm_dir"], config["plot_dir"], local_dir=eval_setup['local_dir'], sel_groups_train=train_setup['groups'], sel_groups_test=eval_setup['groups'], file_suffix=train_setup['suffix'])

                            if len(groups_train) == 1:
                                group = groups_train[0]
                                results_cbm.loc[len(results_cbm.index)] = [attr, train_setup['dataset'],
                                                                           eval_setup['dataset'], model, pool, group,
                                                                           group, f1, corr.statistic, corr.pvalue]
                            else:
                                assert len(groups_train) == len(corr.statistic), (
                                            "list of groups and correlation results do not match: %i / %i" % (
                                    len(groups_train), len(corr.statistic)))
                                # save results to dataframe (one row per group)
                                for i, group in enumerate(groups_train):
                                    results_cbm.loc[len(results_cbm.index)] = [attr, train_setup['dataset'],
                                                                               eval_setup['dataset'], model, pool, groups_eval[i],
                                                                               group, f1, corr.statistic[i], corr.pvalue[i]]

            results_cbm.to_csv(results_cbm_path, index=False)


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('cbm_train_eval.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('cbm_train_eval.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    run_cbm_training(config)
    run_cbm_eval(config)

if __name__ == "__main__":
    main(sys.argv[1:])
