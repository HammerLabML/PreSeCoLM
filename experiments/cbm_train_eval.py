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
import scipy
from sklearn.metrics import f1_score

import torch

# local imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader
import models
import plotting
import utils


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


def train_cbm_train_test(emb_train, y_train, g_train, emb_test, y_test, g_test, groups, concept_weights,
                         class_weights, params_file, checkpoint_file, lc_savefile):
    # could use dev set to determine these
    lambda_concept = 0.5
    lr = 1e-4
    epochs = 80

    # check if checkpoint for this CBM exists already
    if os.path.isfile(checkpoint_file) and os.path.isfile(params_file):
        print("cbm checkpoint and parameter file %s, %s already exist" % (checkpoint_file, params_file))
        return
    else:
        print("checkpoint and/ or parameter file not found: %s, %s" % (checkpoint_file, params_file))

    # determine criterion and prepare class labels for torch model; for multi-labels class weights are applied
    print("fit CBM with groups ", groups)

    n_classes, multi_label = utils.get_number_of_classes(y_test)
    if multi_label:
        print("binary (multi) label: use BCEWithLogitsLoss")
        criterion = torch.nn.BCEWithLogitsLoss
    else:
        print("single label, multi class: use CrossEntropyLoss")
        assert len(y_train.shape) == 1 or y_train.shape[1] == 1
        criterion = torch.nn.CrossEntropyLoss
        class_weights = None
        y_train = y_train.flatten().astype('int')
        y_test = y_test.flatten().astype('int')

    # prepare concept labels
    c_train = data_loader.label2onehot(g_train)
    c_test = data_loader.label2onehot(g_test)

    # init model params
    model_params = {'input_size': emb_train.shape[1], 'output_size': n_classes, 'n_learned_concepts': c_train.shape[1],
                    'n_other_concepts': emb_train.shape[1] - c_train.shape[1], 'hidden_size': 300}
    print("train CBM with params:")
    print(model_params)

    cbm = models.CBM(**model_params)
    cbmWrapper = models.CBMWrapper(cbm, batch_size=64, class_weights=class_weights, criterion=criterion,
                                   concept_criterion=torch.nn.BCEWithLogitsLoss, lr=lr, lambda_concept=lambda_concept,
                                   concept_weights=concept_weights)

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

    # plot learning curve for class and concept F1-score
    x = np.arange(len(f1s_c))
    fig, ax = plt.subplots()
    ax.plot(x, f1s_c, label='concept F1')
    ax.plot(x, f1s_p, label='pred F1')
    ax.legend()
    plt.savefig(lc_savefile, bbox_inches='tight')

    # save model
    torch.save(cbm.state_dict(), checkpoint_file)
    with open(params_file, 'wb') as handle:
        pickle.dump(model_params, handle)


def train_cbms(dataset, model, pooling, batch_size, emb_dir, cbm_dir, plot_dir, local_dir=None, sel_groups=None,
               file_suffix=None):  # lambda_concept=0.5, use_concept_weights=False, lr=1e-4, epochs=80
    learning_curve_dir = '%s/cbm_learning_curves/' % plot_dir
    if not os.path.isdir(learning_curve_dir):
        os.makedirs(learning_curve_dir)

    checkpoint_file, params_file = get_cbm_savefile(cbm_dir, dataset, model, pooling, file_suffix)
    lc_savefile = ('%s/%s_%s_%s_%s.png' % (learning_curve_dir, dataset, model.replace('/', '_'), pooling, file_suffix))

    # check if cbm checkpoint exists (does not cover cv datasets)
    if os.path.isfile(checkpoint_file) and os.path.isfile(params_file):
        print("cbm checkpoint and parameter file %s, %s already exist" % (checkpoint_file, params_file))
        return
    else:
        print("checkpoint and/ or parameter file not found: %s, %s" % (checkpoint_file, params_file))

    # get dataset and embeddings
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset, model, pooling, batch_size, local_dir)

    if len(dataset.splits) == 1:
        print("single-split dataset - train CBM for each CV split")
        for fold_id in range(dataset.n_folds):
            data_dict = dataset.get_cv_split(fold_id)
            X_train, emb_train, y_train, g_train, class_weights, group_weights = data_dict['train']
            X_test, emb_test, y_test, g_test, _, _ = data_dict['test']
            g_train, groups, group_weights = utils.filter_group_labels(dataset.group_names, sel_groups, g_train, group_weights)
            g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

            checkpoint_file_fold = checkpoint_file + ('_%iof%i' % (fold_id, dataset.n_folds))
            params_file_fold = params_file.replace('.pickle', '_%iof%i.pickle' % (fold_id, dataset.n_folds))
            lc_savefile_fold = lc_savefile.replace('.png', '_%iof%i.png' % (fold_id, dataset.n_folds))
            train_cbm_train_test(emb_train, y_train, g_train, emb_test, y_test, g_test, sel_groups, group_weights,
                                 class_weights, params_file_fold, checkpoint_file_fold, lc_savefile_fold)
    else:
        # TODO: check later if new datasets use the same split names
        # TODO: what to do with dev split? combine with test split?
        X_train, emb_train, y_train, g_train, class_weights, group_weights = dataset.get_split('train')
        X_test, emb_test, y_test, g_test, _, _ = dataset.get_split('test')
        g_train, groups, group_weights = utils.filter_group_labels(dataset.group_names, sel_groups, g_train, group_weights)
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups, g_test)

        train_cbm_train_test(emb_train, y_train, g_train, emb_test, y_test, g_test, sel_groups, group_weights,
                             class_weights, params_file, checkpoint_file, lc_savefile)

        # TODO: hopefully not necessary anymore:
        # data_loader with binary group labels provide only 1D labels for two groups; adjust label accordingly
        # in this case, group labels will be learned as single label and are converted to multi-label for evaluation
        # if len(groups) == 2 and (g_test.ndim == 1 or g_test.shape[1] == 1):
        #    groups = [groups[0]+'/'+groups[1]]


def eval_cbm_on_test_split(emb_test, g_test, checkpoint_file, params_file, groups_train, groups_test, add_labels, plot_savefile,
                           dataset_train, dataset_test):
    # load model and parameters
    with open(params_file, "rb") as handle:
        model_params = pickle.load(handle)

    cbm = models.CBM(**model_params)
    with open(checkpoint_file, "rb") as handle:
        cbm.load_state_dict(torch.load(checkpoint_file, weights_only=True))

    # wrapper only needed for predict call, most params don't matter
    cbmWrapper = models.CBMWrapper(cbm, batch_size=32, class_weights=None, criterion=torch.nn.BCEWithLogitsLoss,
                                   concept_criterion=torch.nn.BCEWithLogitsLoss, lr=1e-4, lambda_concept=0.5,
                                   concept_weights=None)

    # concepts encoded as onehot
    if len(groups_test) == 1:
        assert np.all((g_test == 0) | (g_test == 1)), "got only 1 label, but g_test contains multi-class label!"
        c_test = g_test
        n_concepts = 1
    else:
        c_test = data_loader.label2onehot(g_test)
        n_concepts = c_test.shape[1]

    # with onehot encoding number of test groups and shape of test concepts should match
    # the number of training groups might be larger
    assert n_concepts <= len(groups_train) and n_concepts == len(groups_test), ("number of concepts (%i) does not match"
                                                                                " number of train (%i) /test groups "
                                                                                "(%i) ") % (n_concepts,
                                                                                            len(groups_train),
                                                                                            len(groups_test))

    # get concept predictions
    _, concepts = cbmWrapper.predict(emb_test)
    c_pred = (np.array(concepts) >= 0).astype(int)  # concept scores are logits!

    # remove model from gpu
    cbm.to('cpu')
    del cbm

    # handle different dimensionality of test and predicted concepts
    if not utils.is1D(c_test) and utils.is1D(c_pred):
        print("convert 1d prediction to onehot")
        c_pred = data_loader.label2onehot(c_pred, minv=int(np.min(c_test)), maxv=int(np.max(c_test)))
        concepts = np.hstack([-concepts, concepts])
        assert len(groups_train) == 2
    if utils.is1D(c_test) and not utils.is1D(c_pred):
        print("convert 1d test labels to onehot")
        c_test = data_loader.label2onehot(c_test)

    # align labels
    c_test, c_pred, concepts, groups_test, groups_train = utils.align_labels(c_test, c_pred, concepts, groups_test,
                                                                             groups_train)

    # add any- and contrastive labels for further eval
    if add_labels:
        c_test_, groups_test_ = utils.add_contrastive_any_labels(c_test, groups_test)
        concepts_, groups_train_ = utils.add_contrastive_any_labels(concepts, groups_train)
    else:
        groups_test_ = groups_test
        groups_train_ = groups_train
        c_test_ = c_test
        concepts_ = concepts

    # evaluate the concept predictions
    # train and test labels are already aligned
    # compute F1-score and Pearson correlation
    f1 = f1_score(c_test, c_pred, average='macro')

    print(f1)
    print(concepts_.shape)
    print(c_test_.shape)
    print(groups_train_)
    print(groups_test_)
    print(dataset_train)
    print(dataset_test)
    if not utils.is1D(concepts_):  # both > 1D
        corr = scipy.stats.pearsonr(concepts_, np.asarray(c_test_))
    else:  # both 1D
        corr = scipy.stats.pearsonr(concepts_.flatten(), c_test_.flatten())

    # plot histograms for all test groups (1 or 0) against all predicted concepts, save plot
    plotting.plot_feature_histogram(concepts_, c_test_, labels=groups_test_, features=groups_train_,
                                    xlabel=dataset_train, ylabel=dataset_test, savefile=plot_savefile)

    return f1, corr, groups_train_, groups_test_


def evaluate_cbms(dataset_train, dataset_test, model, pooling, batch_size, emb_dir, cbm_dir, plot_dir, local_dir=None,
                  local_dir_train=None, sel_groups_train=None, sel_groups_test=None, file_suffix=None):
    # check if 'any' label specified
    add_labels = (sel_groups_test[0] == 'any')
    sel_groups_test_ = [group for group in sel_groups_test if not group == 'any']

    checkpoint_file, params_file = get_cbm_savefile(cbm_dir, dataset_train, model, pooling, '')

    # get the training dataset to determine if it has CV splits
    dataset_tr = data_loader.get_dataset(dataset_train, local_dir_train)
    if len(dataset_tr.splits) == 1:
        file_names = []
        for fold_id in range(dataset_tr.n_folds):
            checkpoint_file_fold = checkpoint_file + ('_%iof%i' % (fold_id, dataset_tr.n_folds))
            params_file_fold = params_file.replace('.pickle', '_%iof%i.pickle' % (fold_id, dataset_tr.n_folds))
            file_names.append((checkpoint_file_fold, params_file_fold))
    else:
        file_names = [(checkpoint_file, params_file)]
    print("cbm checkpoint/param files for train case: ", file_names)

    model_name = model.replace('/', '_')
    plot_dir = '%s/cbm_eval' % plot_dir
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if file_suffix is not None:
        plot_savefile = ('%s/%s_%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling,
                                                    file_suffix))
    else:
        plot_savefile = ('%s/%s_%s_%s_%s.png' % (plot_dir, dataset_train, dataset_test, model_name, pooling))

    # get the eval dataset + embeddings
    dataset = utils.get_dataset_with_embeddings(emb_dir, dataset_test, model, pooling, batch_size, local_dir)

    f1 = None
    r_val = None
    p_val = None
    groups_train_ = None
    groups_test_ = None
    if len(dataset.splits) == 1:
        print("single-split dataset - train CBM for each CV split")
        f1s = []
        rs = []
        ps = []
        fold_missing = False
        for fold_id in range(dataset.n_folds):
            plot_savefile_fold = plot_savefile.replace('.png', '_%iof%i.png' % (fold_id, dataset.n_folds))

            data_dict = dataset.get_cv_split(fold_id)
            X_test, emb_test, y_test, g_test, _, _ = data_dict['test']
            g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups_test_, g_test)

            for i, cur_file_name in enumerate(file_names):
                (checkpoint_file_fold, params_file_fold) = cur_file_name
                plot_savefile_fold_ = plot_savefile_fold.replace('.png', '_%i.png' % i)

                if not os.path.isfile(checkpoint_file_fold):
                    print("could not find CBM checkpoint for %s %s %s" % (dataset_train, model, pooling))
                    print(checkpoint_file)
                    fold_missing = True
                    break

                (cur_f1, cur_corr,
                 groups_train_, groups_test_) = eval_cbm_on_test_split(emb_test, g_test, checkpoint_file_fold,
                                                                       params_file_fold, sel_groups_train,
                                                                       sel_groups_test_, add_labels, plot_savefile_fold_,
                                                                       dataset_train, dataset_test)
                f1s.append(cur_f1)
                rs.append(cur_corr.statistic)
                ps.append(cur_corr.pvalue)

            if fold_missing:
                print("at least one checkpoint of the training data folds is missing, skip %s" % dataset_train)
                print(file_names)
                return

        f1 = np.mean(f1s)
        # corr might be a list of correlations per group
        r_val = np.mean(rs, axis=0)
        p_val = np.mean(ps, axis=0)
        if type(rs[0]) is not list and not type(rs[0]) is np.ndarray:
            print("after %i folds of crossval got F1 = %.2f +/- %.2f, R = %.2f +/- %.2f, "
                  "p = %.2f +/- %.2f" % (dataset.n_folds, f1, np.std(f1s), r_val, np.std(rs), p_val, np.std(ps)))
    else:
        # TODO: check later if new datasets use the same split names
        # TODO: what to do with dev split? combine with test split?
        X_test, emb_test, y_test, g_test, _, _ = dataset.get_split('test')
        g_test, _, _ = utils.filter_group_labels(dataset.group_names, sel_groups_test_, g_test)

        f1s = []
        rs = []
        ps = []
        for i, cur_file_name in enumerate(file_names):  # might need to iterate through cv splits of training dataset
            (checkpoint_file_fold, params_file_fold) = cur_file_name
            if not os.path.isfile(checkpoint_file_fold):
                print("could not find CBM checkpoint for %s %s %s" % (dataset_train, model, pooling))
                print(checkpoint_file_fold)
                print("since at least one checkpoint is missing, skip eval for %s %s %s" % (dataset_train, model, pooling))
                return

            plot_savefile_fold = plot_savefile.replace('.png', '_%i.png' % i)
            cur_f1, cur_corr, groups_train_, groups_test_ = eval_cbm_on_test_split(emb_test, g_test,
                                                                                   checkpoint_file_fold,
                                                                                   params_file_fold, sel_groups_train,
                                                                                   sel_groups_test_, add_labels,
                                                                                   plot_savefile_fold, dataset_train,
                                                                                   dataset_test)
            f1s.append(cur_f1)
            rs.append(cur_corr.statistic)
            ps.append(cur_corr.pvalue)

        f1 = np.mean(f1s)
        # corr might be a list of correlations per group
        r_val = np.mean(rs, axis=0)
        p_val = np.mean(ps, axis=0)
        if type(rs[0]) is not list and not type(rs[0]) is np.ndarray:
            print("after %i folds of crossval got F1 = %.2f +/- %.2f, R = %.2f +/- %.2f, "
                  "p = %.2f +/- %.2f" % (dataset.n_folds, f1, np.std(f1s), r_val, np.std(rs), p_val, np.std(ps)))

    return f1, r_val, p_val, groups_train_, groups_test_


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
                if setup['dataset'] in ['twitterAAE']:
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
                    if train_setup['dataset'] in ['twitterAAE']:
                        continue

                    # check if train dataset relevant for the current attribute
                    dataset_in_test_case = False
                    for test_setup in eval_setups:
                        if test_setup['dataset'] == train_setup['dataset']:
                            dataset_in_test_case = True
                    if not dataset_in_test_case:
                        continue
                    
                    for eval_setup in eval_setups:
                            
                        # skip 'same dataset' cases where no training data is available
                        #if train_setup['dataset'] == eval_setup['dataset'] and eval_setup['dataset'] in ['twitterAAE', 'crows_pairs']:
                        #    continue  # no training data available for twitterAAE and crowspairs (TODO CV on test data)

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
                            f1, r_val, p_val, groups_train, groups_eval = evaluate_cbms(train_setup['dataset'], eval_setup['dataset'], model, pool,
                                                          batch_size, config["embedding_dir"], config["cbm_dir"],
                                                          config["plot_dir"], local_dir=eval_setup['local_dir'],
                                                          local_dir_train=train_setup['local_dir'],
                                                          sel_groups_train=train_setup['groups'],
                                                          sel_groups_test=eval_setup['groups'],
                                                          file_suffix=train_setup['suffix'])

                            if len(groups_train) == 1:
                                group = groups_train[0]
                                results_cbm.loc[len(results_cbm.index)] = [attr, train_setup['dataset'],
                                                                           eval_setup['dataset'], model, pool, group,
                                                                           group, f1, r_val, p_val]
                            else:
                                assert len(groups_train) == len(r_val), (
                                            "list of groups and correlation results do not match: %i / %i" % (
                                                len(groups_train), len(r_val)))
                                # save results to dataframe (one row per group)
                                for i, group in enumerate(groups_train):
                                    results_cbm.loc[len(results_cbm.index)] = [attr, train_setup['dataset'],
                                                                               eval_setup['dataset'], model, pool, groups_eval[i],
                                                                               group, f1, r_val[i], p_val[i]]

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

    #run_cbm_training(config)
    run_cbm_eval(config)


if __name__ == "__main__":
    main(sys.argv[1:])
