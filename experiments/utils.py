import numpy as np
import random
import scipy
import math
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from sentence_transformers import SentenceTransformer
import torch


# local import for data loader
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader
import models

SUPPORTED_OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
SUPPORTED_SENTENCE_TRANSFORMER = ["Qwen/Qwen3-Embedding-0.6B"]
SUPPORTED_HUGGINGFACE_MODELS = ["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased",
                                "google/electra-small-generator", "google/electra-base-generator", "google/electra-large-generator",
                                "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2",
                                "microsoft/deberta-v3-small", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large",
                                "gpt2", "gpt2-large", "distilgpt2", "gpt2-medium", "openai-gpt",
                                "roberta-base", "roberta-large", "distilroberta-base",
                                "google/electra-small-discriminator", "google/electra-base-discriminator", "google/electra-large-discriminator",
                                "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
                                "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
                                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/deepseek-llm-7b-base", "deepseek-ai/deepseek-llm-7b-chat",
                                "facebook/bart-base", "facebook/bart-large", "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
                                "google-t5/t5-small", "google-t5/t5-base",
                                "xlnet/xlnet-base-cased", "xlnet/xlnet-large-cased"]
# possible other models (test):
# "openai-gpt"
# "xlm-roberta-base", "xlm-roberta-large",
# "bert-large-uncased-whole-word-masking", "bert-base-multilingual-uncased",
# "distilbert-base-multilingual-cased", "distilbert-base-uncased-finetuned-sst-2-english",
# "roberta-large-openai-detector", "roberta-base-openai-detector",
# "xlnet-base-cased", "xlnet-large-cased",

# TODO: move to some config
#LABEL_MATCHES = {'black': 'aa', 'aa': 'black', 'M': 'male', 'male': 'M', 'F': 'female', 'female': 'F',
#                 'homosexual_gay_or_lesbian': 'homosexual', 'homosexual': 'homosexual_gay_or_lesbian',
#                 'psychiatric_or_mental_illness': 'mental_disability_illness', 'mental_disability_illness': 'psychiatric_or_mental_illness',
#                 'european/white': 'white', 'white': 'european/white', 'african/black': 'black', 'black': 'african/black', 'south east asian': 'asian', 'asian': 'south east asian',
#                 'aae': 'black', 'neutral (ae_dialect)': 'white', 'latino': 'hispanic'}
LABEL_MATCHES = {'M': 'male', 'F': 'female',
                 'aa': 'black', 'african/black': 'black', 'aae': 'black', 'neutral (ae_dialect)': 'white', 'european/white': 'white', 'south east asian': 'asian', 'latino': 'hispanic',
                 'homosexual_gay_or_lesbian': 'homosexual',
                 'psychiatric_or_mental_illness': 'mental_disability_illness'}


def defining_terms_labels_from_dict(term_dict: dict) -> (list, np.ndarray):
    terms = []
    lbl = []
    cur_lbl = 1
    for key, cur_terms in term_dict.items():
        terms += cur_terms
        if key == 'neutral':
            lbl += [0 for t in cur_terms]
        else:
            lbl += [cur_lbl for t in cur_terms]
            cur_lbl += 1

    return terms, np.asarray(lbl)


def select_def_terms_for_dataset(def_terms_per_attr: dict, group_matches: dict):
    groups = []
    for k, v in group_matches.items():
        groups += v

    new_dict = {}
    for attr, terms_per_group in def_terms_per_attr.items():
        need_attr = False
        for group, def_terms in terms_per_group.items():
            if group in groups:
                need_attr = True
        if need_attr:
            new_dict.update({attr: terms_per_group})

    return new_dict


def get_multi_attr_def_terms_labels(term_dicts: dict) -> (list, np.ndarray):
    n_attr = len(term_dicts)
    cur_col = 0

    terms_stacked = []
    lbl_stacked = []
    group_lbl = []
    for attr, td in term_dicts.items():
        terms, lbl = defining_terms_labels_from_dict(td)

        # transform single-label into multi-label arr (set one column with current lbl, everything else as -1)
        lbl_resized = np.ones((len(terms), n_attr)) * -1
        lbl_resized[:, cur_col] = lbl

        terms_stacked += terms
        lbl_stacked.append(lbl_resized)
        cur_col += 1

        cur_groups = []
        for key in td.keys():
            if key != 'neutral':
                cur_groups.append(key)
        group_lbl.append(cur_groups)

    lbl_stacked = np.vstack(lbl_stacked)

    # count total number of labels (excl. -1)
    n_lbl = 0
    for groups in group_lbl:
        if len(groups) == 1:  # only 1D bias space
            n_lbl += 1
        else:
            n_lbl += len(groups)+1  # on for each group + any dir

    return terms_stacked, lbl_stacked, n_lbl, group_lbl


def add_contrastive_any_labels(g_test, group_names):
    n_groups = g_test.shape[1]
    new_shape = (g_test.shape[0], n_groups * 2 + 1)

    g_true = np.zeros(new_shape)
    g_true[:, 0] = g_test.any(axis=1)  # any group
    for i in range(n_groups):
        # group i - other groups
        g_true[:, i + 1] = g_test[:, i] - np.sum(np.delete(g_test, i, 1), axis=1)
    g_true[:, -n_groups:] = g_test

    new_group_names = ['any'] + [name + ' vs. rest' for name in group_names] + group_names

    return g_true, new_group_names


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
    return np.matmul(X, T.T)


def align_labels(y_test, y_pred, pred, label_test, label_pred):

    # align predictions and test labels:
    # there might be more predicted groups than test groups
    # the order of groups might be different
    # some data_loader use different labels for the same group
    print("labels before alignment:")
    print(label_test)
    print(label_pred)

    # convert non-default labels to allow matching of test/pred labels
    label_test = [LABEL_MATCHES[lbl] if lbl in LABEL_MATCHES.keys() else lbl for lbl in label_test]
    label_pred = [LABEL_MATCHES[lbl] if lbl in LABEL_MATCHES.keys() else lbl for lbl in label_pred]

    print("labels converted to default names:")
    print(label_test)
    print(label_pred)

    label_shared = [lbl for lbl in label_test if lbl in label_pred]
    ids_pred = [label_pred.index(lbl) for lbl in label_shared]
    ids_test = [label_test.index(lbl) for lbl in label_shared]
    y_pred = y_pred[:, ids_pred]
    pred = pred[:, ids_pred]
    y_test = y_test[:, ids_test]

    print("labels after alignment:")
    print(label_shared)
    print(label_shared)

    print(y_pred.shape)
    print(y_test.shape)

    return y_test, y_pred, pred, label_shared, label_shared


def get_model_type_architecture(model_name):
    if model_name in SUPPORTED_OPENAI_MODELS:
        return 'text-embedding-3', 'embedder'
    elif model_name in SUPPORTED_SENTENCE_TRANSFORMER:
        return 'sentence_transformer', 'sentence_transformer' # TODO
    else:
        assert model_name in SUPPORTED_HUGGINGFACE_MODELS, "model '%s' is not among the supported openai or huggingface models!" % model_name
        lm = models.get_pretrained_model(model_name, 2, batch_size=1)
        architecture = 'encoder'
        if lm.model.config.is_encoder_decoder:
            architecture = 'encoder-decoder'
        elif 'gpt' in model_name or 'xlnet' in model_name or 'opt' in model_name or 'llama' in model_name:
            architecture = 'decoder'

        model_family = 'unknown'
        if 'pythia' in model_name:
            model_family = 'pythia'
        elif 'deepseek' in model_name:
            model_family = 'deepseek'
        elif 'model_type' in lm.model.config.__dict__.keys():
            model_family = lm.model.config.model_type

        return model_family, architecture


def get_dataset_with_embeddings(emb_dir: str, dataset_name: str, model_name: str, pooling: str, batch_size: int,
                                local_dir=None, defining_term_dict=None):
    assert (model_name in SUPPORTED_OPENAI_MODELS or model_name in SUPPORTED_HUGGINGFACE_MODELS 
            or model_name in SUPPORTED_SENTENCE_TRANSFORMER), "model '%s' is not among the supported openai or huggingface models!" % model_name

    # load dataset
    dataset = data_loader.get_dataset(dataset_name, local_dir)

    # embed dataset and defining terms if given
    emb_defining_attr_dict = None
    if model_name in SUPPORTED_OPENAI_MODELS:
        # try to load pre-computed embeddings of openai model
        print("using openai models -> load precomputed embeddings...")
        split_emb = {}
        for split in dataset.splits:
            data, _, _, _, _, _ = dataset.get_split(split)
            split_emb[split] = models.get_embeddings(data, dataset_name, split, model_name, emb_dir)
        dataset.set_preprocessed_data(split_emb)

        if defining_term_dict is not None:
            emb_defining_attr_dict = models.get_defining_term_embeddings(defining_term_dict, model_name, emb_dir)
    elif model_name in SUPPORTED_SENTENCE_TRANSFORMER:
        print("sentence transformer...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        lm = SentenceTransformer(model_name, device=device)

        split_emb = {}
        for split in dataset.splits:
            print("embed %s split..." % split)
            data, _, _, _, _, _ = dataset.get_split(split)
            split_emb[split] = models.load_or_compute_embeddings(data, lm, dataset_name, split, emb_dir, is_sentence_transformer=True, 
                                                                 batch_size=batch_size, model_name=model_name)
        dataset.set_preprocessed_data(split_emb)

        if defining_term_dict is not None:
            if isinstance(defining_term_dict, dict):
                # defining terms might be passed as dictionary (sorted by protected groups)
                if defining_term_dict is not None:
                    emb_defining_attr_dict = {attr: {} for attr in defining_term_dict.keys()}
                    # defining terms is a list of defining attr for different attributes (list[list[list]])
                    print("embed defining terms...")
                    for attr, dterm_dict in defining_term_dict.items():
                        for group, dterms in dterm_dict.items():
                            emb_defining_attr_dict[attr][group] = lm.encode(dterms, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
            else:
                # might also be a list, simply embed then
                assert isinstance(defining_term_dict, list), "expected list or dictionary with defining terms"
                emb_defining_attr_dict = lm.encode(defining_term_dict, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
                
    else:  # model_name in SUPPORTED_HUGGINGFACE_MODELS (already checked by assert)
        print("huggingface model...")
        # load huggingface model and get embeddings (either loaded or computed)
        if dataset.n_classes == 0:  # twitterAAE does not have labels
            n_classes = 2
        else:
            n_classes = dataset.n_classes
        lm = models.get_pretrained_model(model_name, n_classes, batch_size=batch_size, pooling=pooling,
                                         multi_label=dataset.multi_label)

        split_emb = {}
        for split in dataset.splits:
            print("embed %s split..." % split)
            data, _, _, _, _, _ = dataset.get_split(split)
            split_emb[split] = models.load_or_compute_embeddings(data, lm, dataset_name, split, emb_dir)
        dataset.set_preprocessed_data(split_emb)

        if defining_term_dict is not None:
            if isinstance(defining_term_dict, dict):
                # defining terms might be passed as dictionary (sorted by protected groups)
                if defining_term_dict is not None:
                    emb_defining_attr_dict = {attr: {} for attr in defining_term_dict.keys()}
                    # defining terms is a list of defining attr for different attributes (list[list[list]])
                    print("embed defining terms...")
                    for attr, dterm_dict in defining_term_dict.items():
                        for group, dterms in dterm_dict.items():
                            emb_defining_attr_dict[attr][group] = lm.embed(dterms)
            else:
                # might also be a list, simply embed then
                assert isinstance(defining_term_dict, list), "expected list or dictionary with defining terms"
                emb_defining_attr_dict = lm.embed(defining_term_dict)
            
        lm.model.to('cpu')
        del lm

    if defining_term_dict is not None:
        return dataset, emb_defining_attr_dict
    else:
        return dataset


def filter_group_labels(all_groups: list, sel_groups: list, group_lbl: np.ndarray, group_weights: np.ndarray = None):
    if sel_groups is None or sel_groups == all_groups:
        return group_lbl, sel_groups, group_weights

    msg = "group_lbl are either singe-label or do not match the number of groups"
    assert type(group_lbl) is np.ndarray and group_lbl.ndim > 1 and group_lbl.shape[1] == len(all_groups), msg

    #print("filter group_lbl from ", all_groups, " to ", sel_groups)
    filter_ids = [all_groups.index(group) for group in sel_groups]
    group_lbl = np.squeeze(group_lbl[:, [filter_ids]])

    if group_lbl.ndim == 1:
        group_lbl = group_lbl.reshape(-1, 1)

    msg = "group_lbl does not match the expected shape of [%i,%i], got %s instead" % (group_lbl.shape[0],
                                                                                      len(sel_groups), group_lbl.shape)
    assert type(group_lbl) is np.ndarray and group_lbl.shape[1] == len(sel_groups), msg

    groups = [all_groups[idx] for idx in filter_ids]
    if sel_groups is not None:
        assert sel_groups == groups, "expected these groups: %s, but after filtering got: %s" % (sel_groups, groups)

    if group_weights is not None:
        group_weights = group_weights[filter_ids]

    return group_lbl, groups, group_weights


"""
def get_dataset_and_embeddings(emb_dir: str, dataset: str, model_name: str, pooling: str, batch_size: int,
                               local_dir=None, defining_term_dict=None, sel_groups=None):
    assert (model_name in SUPPORTED_OPENAI_MODELS or model_name in SUPPORTED_HUGGINGFACE_MODELS), "model '%s' is not among the supported openai or huggingface models!" % model_name

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
            g_train = np.squeeze(g_train[:, [filter_ids]])
        g_test = np.squeeze(g_test[:, [filter_ids]])
    else:
        # convert to ndarray
        if type(g_test) == list:
            g_test = np.asarray(g_test)
            if len(g_train) > 0:
                g_train = np.asarray(g_train)

        if sel_groups is None:
            sel_groups = protected_attr_dict['labels']

    # embed dataset and defining terms if given
    emb_defining_attr_dict = None
    emb_train = []
    if model_name in SUPPORTED_OPENAI_MODELS:
        # try to load pre-computed embeddings of openai model
        print("using openai models -> load precomputed embeddings...")
        emb_test = models.get_embeddings(X_test, dataset, 'test', model_name, emb_dir)
        if len(X_train) > 0:
            emb_train = models.get_embeddings(X_train, dataset, 'train', model_name, emb_dir)

        if defining_term_dict is not None:
            emb_defining_attr_dict = models.get_defining_term_embeddings(defining_term_dict, model_name, emb_dir)
    else:  # model_name in SUPPORTED_HUGGINGFACE_MODELS (because assert earlier!)
        # load huggingface model and get embeddings (either loaded or computed)
        lm = models.get_pretrained_model(model_name, n_classes, batch_size=batch_size, pooling=pooling, multi_label=multi_label)

        emb_test = models.load_or_compute_embeddings(X_test, lm, dataset, 'test', emb_dir)
        if len(X_train) > 0:
            emb_train = models.load_or_compute_embeddings(X_train, lm, dataset, 'train', emb_dir)

        if defining_term_dict is not None:
            emb_defining_attr_dict = {attr: {} for attr in defining_term_dict.keys()}
            # defining terms is a list of defining attr for different attributes (list[list[list]])
            print("embed defining terms...")
            for attr, dterm_dict in defining_term_dict.items():
                for group, dterms in dterm_dict.items():
                    emb_defining_attr_dict[attr][group] = lm.embed(dterms)
        lm.model.to('cpu')
        del lm

    return X_train, emb_train, y_train, g_train, X_test, emb_test, y_test, g_test, sel_groups, emb_defining_attr_dict, class_weights


"""
