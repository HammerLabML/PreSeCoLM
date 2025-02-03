import numpy as np
import random
import scipy
import math
from sklearn.metrics import f1_score, precision_score, recall_score

from pie_data import get_dataset, label2onehot
from pie.helper import normalize
from pie import PIEmbedder, compute_multi_attr_bias_space, get_bias_space, get_feature2emb_transform, get_emb2feature_transform, get_orthonormal_space
from models import CBM, CBMWrapper, CAV, load_or_compute_embeddings, get_pretrained_model, get_pretrained_model_with_batch_size_lookup, get_finetuned_model, get_embeddings, get_defining_term_embeddings, ClfWrapper, Classifier, BertLikeClassifier, LinearClassifier

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
                 'psychiatric_or_mental_illness': 'mental disability, illness', 'mental disability, illness': 'psychiatric_or_mental_illness'}

def cossim(x: np.ndarray, y: np.ndarray):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))

def feature_label_corr(X: np.ndarray, y: list, groups: list, feature_label: list):
    assert len(X.shape) == 2
    assert X.shape[1] == len(feature_label)
    if type(y) == list:
        y = np.asarray(y)

    if len(y.shape) == 1:
        assert len(groups) == 2
        identity_label = ('%s/%s' % (groups[0],groups[1]))
        corr_res = {identity_label: {}}
        # single binary identity label
        for feature_id in range(X.shape[1]):
            res = scipy.stats.pearsonr(X[:,feature_id], y)
            print("identity label (%s) and feature %i (%s) got r=%.3f and p=%.3f" % (identity_label, feature_id, feature_label[feature_id], res.statistic, res.pvalue))
            corr_res[identity_label].update({feature_label[feature_id]: res})
    else:
        # multiple binary identity labels
        corr_res = {group: {} for group in groups}
        for i, group in enumerate(groups):
            for feature_id in range(X.shape[1]):
                res = scipy.stats.pearsonr(X[:,feature_id], y[:,i])
                print("%s and feature %i (%s) got r=%.3f and p=%.3f" % (group, feature_id, feature_label[feature_id], res.statistic, res.pvalue))
                corr_res[group].update({feature_label[feature_id]: res})

    return corr_res

def feature_label_corr_1D(X: np.ndarray, y: list, groups: list):
    assert len(X.shape) == 1
    if type(y) == list:
        y = np.asarray(y)

    if len(y.shape) == 1:
        assert len(groups) == 2
        identity_label = ('%s/%s' % (groups[0],groups[1]))
        corr_res = {identity_label: {}}
        # single binary identity label
        res = scipy.stats.pearsonr(X, y)
        #print("identity label (%s) and feature got r=%.3f and p=%.3f" % (identity_label, res.statistic, res.pvalue))
        corr_res[identity_label] = res
    else:
        # multiple binary identity labels
        corr_res = {group: {} for group in groups}
        for i, group in enumerate(groups):
            res = scipy.stats.pearsonr(X, y[:,i])
            #print("%s and feature got r=%.3f and p=%.3f" % (group, res.statistic, res.pvalue))
            corr_res[group] = res

    return corr_res

def feature_corr(X: np.ndarray, T: np.ndarray, y: list, groups: list, feature_label: list):
    if len(groups) == 2: # single-label
        identity_label = ('%s/%s' % (groups[0],groups[1]))
        x0 = np.asarray([X[i,:] for i in range(len(X)) if y[i] == 0])
        x1 = np.asarray([X[i,:] for i in range(len(X)) if y[i] == 1])
        features_by_data = np.mean(x1, axis=0)-np.mean(x0, axis=0)
        median_feature = np.median(x1, axis=0)-np.median(x0, axis=0)

        similarities = []
        for i in range(T.shape[0]):
            sim = cossim(features_by_data.flatten(), T[i,:].flatten())
            similarities.append(sim)
        
        print("similarities with %s per feature:" % identity_label)
        for i in range(len(similarities)):
            print("%s: %.3f" % (feature_label[i], similarities[i]))
        
    else: # multiple identity labels (one per group)
        features_by_data = np.zeros((X.shape[0], len(groups)))
        similarities = []
        for k in range(len(groups)):
            identity_label = groups[k]
            x0 = np.asarray([X[i,:] for i in range(len(X)) if y[i,k] == 0])
            x1 = np.asarray([X[i,:] for i in range(len(X)) if y[i,k] == 1])
            features_by_data = np.mean(x1, axis=0)-np.mean(x0, axis=0)
            similarities_k = []
            for i in range(T.shape[0]):
                sim = cossim(features_by_data.flatten(), T[i,:].flatten())
                similarities_k.append(sim)
                
            print("similarities with %s per feature:" % identity_label)
            print(similarities_k)
            #for i in range(len(similarities_k)):
            #    print("%s: %.3f" % (feature_label[i], similarities_k[i]))
        similarities.append(similarities_k)
        
    return similarities

def test_classification(X_train, X_test, y_train, y_test):
    classifier = [(LogisticRegression(), 'Logistic Regression'), (LinearSVC(), 'Linear SVM'), (SVC(), 'SVM (RBF Kernel)'), 
                  (KNeighborsClassifier(), 'KNN'), (DecisionTreeClassifier(), 'Decision Tree')]
    results = {}
    for clf, name in classifier:
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.update({name: f1})
    return results

def test_feature_reconstruction(X_train, X_test, y_train, y_test, feature_label):
    # baseline all features
    results = {}
    results['baseline'] = test_classification(X_train, X_test, y_train, y_test)
    print("baseline results: ")
    print(results['baseline'])

    for i, feature in enumerate(feature_label):
        X_train_ = np.delete(X_train, [i], axis=1)
        X_test_ = np.delete(X_test, [i], axis=1)
        results[feature] = test_classification(X_train_, X_test_, y_train, y_test)
        print(feature+" results: ")
        print(results[feature])
    return results

def get_features(X, T):
    return np.matmul(X,T.T)

def compute_bias_space_robustness(emb, B, repeat = 1000):
    mean_angles = []
    std_angles = []
    ns = range(2,emb.shape[1])
    for n_terms in ns:
        angles = []
        for i in range(repeat):
            ids = random.sample(range(emb.shape[1]), n_terms)
            B_ = get_bias_space(emb[:,ids,:])

            if B.shape[0] > 1:
                angles.append(np.sum([[cossim(B[i,:].flatten(), B_[j,:].flatten()) for i in range(B.shape[0])] for j in range(B_.shape[0])]))
            else:
                angles.append(cossim(B.flatten(),B_.flatten()))

        mean_angles.append(np.mean(angles))
        std_angles.append(np.std(angles))

    return np.asarray(ns), np.asarray(mean_angles), np.asarray(std_angles)

def load_dataset_and_get_finetuned_model(model_name, dataset_name, batch_size_lookup, pooling='mean', epochs=5, local_dir=None, run_eval=False):
    X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, _ = get_dataset(dataset_name, local_dir=local_dir)
    lm = get_finetuned_model(model_name, dataset_name, batch_size_lookup, n_classes, multi_label, X_train, y_train, X_test, y_test, pooling=pooling, epochs=epochs, run_eval=run_eval)
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
        X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attr_dict = get_dataset(dataset, local_dir=local_dir)
    else:
        X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attr_dict = get_dataset(dataset)
    g_train = protected_attr_dict['train']
    g_test = protected_attr_dict['test']
    #groups = protected_attr_dict['labels']

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
        emb_test = get_embeddings(X_test, dataset, 'test', model_name, emb_dir)
        if len(X_train) > 0:
            emb_train = get_embeddings(X_train, dataset, 'train', model_name, emb_dir)

        if defining_terms is not None:
            emb_defining_attr = get_defining_term_embeddings(defining_terms, model_name, emb_dir)
    elif model_name in SUPPORTED_HUGGINGFACE_MODELS:
        # load huggingface model and get embeddings (either loaded or computed)
        lm = get_pretrained_model(model_name, n_classes, batch_size=batch_size, pooling=pooling, multi_label=multi_label)

        emb_test = load_or_compute_embeddings(X_test, lm, dataset, 'test', emb_dir)
        if len(X_train) > 0:
            emb_train = load_or_compute_embeddings(X_train, lm, dataset, 'train', emb_dir)

        if defining_terms is not None:
            # defining terms is a list of defining attr for different attributes (list[list[list]])
            print("embed defining terms...")
            emb_defining_attr = [np.asarray([lm.embed(attr) for attr in terms]) for terms in defining_terms]

        lm.model.to('cpu')
        del lm
    else:
         print("error: model %s not among the supported openai and huggingface models")

    return X_train, emb_train, y_train, g_train, X_test, emb_test, y_test, g_test, sel_groups, emb_defining_attr, class_weights


