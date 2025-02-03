import numpy as np
import random
import scipy
import math
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

# local import for data loader
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader
import models

SUPPORTED_OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
SUPPORTED_HUGGINGFACE_MODELS = ["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased",
                                "google/electra-small-generator", "google/electra-base-generator", "google/electra-large-generator",
                                "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2",
                                "gpt2", "gpt2-large", "distilgpt2",
                                "roberta-base", "roberta-large", "distilroberta-base",
                                "google/electra-small-discriminator", "google/electra-base-discriminator", "google/electra-large-discriminator"]
# possible other models (test):
# "openai-gpt"
# "xlm-roberta-base", "xlm-roberta-large",
# "bert-large-uncased-whole-word-masking", "bert-base-multilingual-uncased",
# "distilbert-base-multilingual-cased", "distilbert-base-uncased-finetuned-sst-2-english",
# "roberta-large-openai-detector", "roberta-base-openai-detector",
# "xlnet-base-cased", "xlnet-large-cased",

# TODO: move to some config
LABEL_MATCHES = {'black': 'aa', 'aa': 'black', 'M': 'male', 'male': 'M', 'F': 'female', 'female': 'F',
                 'homosexual_gay_or_lesbian': 'homosexual', 'homosexual': 'homosexual_gay_or_lesbian',
                 'psychiatric_or_mental_illness': 'mental_disability_illness', 'mental_disability_illness': 'psychiatric_or_mental_illness'}


def load_dataset_and_get_finetuned_model(model_name, dataset_name, batch_size_lookup, pooling='mean', epochs=5, local_dir=None, run_eval=False):
    X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, _ = data_loader.get_dataset(dataset_name, local_dir=local_dir)
    lm = models.get_finetuned_model(model_name, dataset_name, batch_size_lookup, n_classes, multi_label, X_train, y_train, X_test, y_test, pooling=pooling, epochs=epochs, run_eval=run_eval)
    return lm


def is1D(x):
    if type(x) == list:
        x = np.asarray(x)
    return x.ndim==1 or x.shape[1] == 1


def is_onehot(y):
    return type(y) == np.ndarray and y.ndim >= 2 and y.shape[1] > 1 and np.min(y) == 0 and np.max(y) == 1


def get_number_of_classes(y):
    multi_label = is_onehot(y)
    if multi_label:
        n_classes = y.shape[1]
    else:
        n_classes = int(np.max(y)+1)

    return n_classes, multi_label


def get_features(X, T):
    return np.matmul(X,T.T)


def eval_with_label_match(y_test, y_pred, pred, label_test, label_pred, average='weighted'):
    # align predictions and test labels:
    # there might be more predicted groups than test groups
    # the order of groups might be different
    # some data_loader use different labels for the same group
    if len(label_test) < len(label_pred):
        # got predictions for more groups than available in test data
        # reduce pred to match the number of test groups (minding the order of groups!)
        if len(label_test) == 1 and y_test.shape[1] == 2:
            y_test = y_test[:,1]
        sel_labels = [lbl if lbl in label_pred else LABEL_MATCHES[lbl] for lbl in label_test]
        ids_pred = [label_pred.index(lbl) for lbl in sel_labels]
        y_pred = y_pred[:,ids_pred]
        pred = pred[:,ids_pred]
    else:
        # same groups in pred in test labels; check order and match inconsistent names that refer to the same group
        sel_labels = [lbl if lbl in label_test else LABEL_MATCHES[lbl] for lbl in label_pred]
        if y_test.ndim > 1:
            ids_test = [label_test.index(lbl) for lbl in sel_labels]
            y_test = y_test[:, ids_test]

    if not is1D(pred): # both > 1D
        cav_corr = scipy.stats.pearsonr(pred, np.asarray(y_test))
    else: # both 1D
        cav_corr = scipy.stats.pearsonr(pred.flatten(), y_test.flatten())

    return f1_score(y_test, y_pred, average=average), cav_corr, sel_labels


def get_dataset_and_embeddings(emb_dir: str, dataset: str, model_name: str, pooling: str, batch_size: int, local_dir=None, defining_terms=None, sel_groups=None):
    # load the dataset
    if local_dir is not None:
        X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attr_dict = data_loader.get_dataset(dataset, local_dir=local_dir)
    else:
        X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attr_dict = data_loader.get_dataset(dataset)
    g_train = protected_attr_dict['train']
    g_test = protected_attr_dict['test']
    # groups = protected_attr_dict['labels']

    # preprocess group labels
    if sel_groups is not None and type(g_test) == np.ndarray and g_test.ndim > 1 and g_test.shape[1] > 1:
        # if multi-label and a selection of groups specified, reduce labels to these groups
        filter_ids = [protected_attr_dict['labels'].index(group) for group in sel_groups]
        if (type(g_train) == np.ndarray) or g_train != []:
            g_train = np.squeeze(g_train[:,[filter_ids]])
        g_test = np.squeeze(g_test[:,[filter_ids]])
    else:
        # convert to ndarray
        if type(g_test) == list:
            g_test = np.asarray(g_test)
            if len(g_train) > 0:
                g_train = np.asarray(g_train)

        if sel_groups is None:
            sel_groups = protected_attr_dict['labels']

    # embed dataset and defining terms if given
    emb_defining_attr = None
    emb_train = []
    if model_name in SUPPORTED_OPENAI_MODELS:
        # try to load pre-computed embeddings of openai model
        print("using openai models -> load precomputed embeddings...")
        emb_test = models.get_embeddings(X_test, dataset, 'test', model_name, emb_dir)
        if len(X_train) > 0:
            emb_train = models.get_embeddings(X_train, dataset, 'train', model_name, emb_dir)

        if defining_terms is not None:
            emb_defining_attr = models.get_defining_term_embeddings(defining_terms, model_name, emb_dir)
    elif model_name in SUPPORTED_HUGGINGFACE_MODELS:
        # load huggingface model and get embeddings (either loaded or computed)
        lm = models.get_pretrained_model(model_name, n_classes, batch_size=batch_size, pooling=pooling, multi_label=multi_label)

        emb_test = models.load_or_compute_embeddings(X_test, lm, dataset, 'test', emb_dir)
        if len(X_train) > 0:
            emb_train = models.load_or_compute_embeddings(X_train, lm, dataset, 'train', emb_dir)

        if defining_terms is not None:
            # defining terms is a list of defining attr for different attributes (list[list[list]])
            print("embed defining terms...")
            emb_defining_attr = [np.asarray([lm.embed(attr) for attr in terms]) for terms in defining_terms]

        lm.model.to('cpu')
        del lm
    else:
        print("error: model %s not among the supported openai and huggingface models")

    return X_train, emb_train, y_train, g_train, X_test, emb_test, y_test, g_test, sel_groups, emb_defining_attr, class_weights


