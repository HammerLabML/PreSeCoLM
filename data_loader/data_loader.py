import numpy as np
import pickle

from datasets import load_dataset
from .bios import filter_bios_dataset
from .crowspairs import preprocess_crowspairs


def is_onehot(y):
    return type(y) == np.ndarray and y.ndim >= 2 and y.shape[1] > 1 and np.min(y) == 0 and np.max(y) == 1


def label2onehot(y, minv=0, maxv=None):
    if is_onehot(y):
        return y
    if type(y) == list:
        y = np.asarray(y, dtype='int')
    else:
        y = y.astype('int')
    if maxv is None:
        maxv = max(1, int(np.max(y)))
    onehot = np.zeros((len(y), 1 + maxv - minv))
    onehot[np.arange(len(y)), y] = 1
    return onehot.astype('float')


def compute_class_weights(y_train: np.ndarray, classes: list):
    # compute positive class weights (based on training data)
    samples_per_class = {lbl: np.sum(y_train[:, i]) for i, lbl in enumerate(classes)}
    print(samples_per_class)
    n_samples = y_train.shape[0]
    # relative weight of 1s per class (compared to 0s not other classes!)
    class_weights = np.asarray([((n_samples - samples_per_class[lbl]) / samples_per_class[lbl]) for lbl in
                                classes])

    for lbl in classes:  # need to verify after filtering!
        assert samples_per_class[lbl] > 0

    print(class_weights)

    return class_weights


def get_dataset(dataset_name, local_dir=None):
    protected_attributes = {'train': None, 'test': None, 'labels': []}

    # GLUE
    if dataset_name in ["sst2"]:  # TODO
        ds = load_dataset("nyu-mll/glue", dataset_name)
        X_train = ds['train']['sentence']
        y_train = ds['train']['label']
        X_test = ds['validation']['sentence']
        y_test = ds['validation']['label']
        class_weights = None
    elif "jigsaw" in dataset_name:
        ds = load_dataset("jigsaw_unintended_bias", data_dir=local_dir, trust_remote_code=True)
        if dataset_name == "jigsaw-multi":
            toxicity_classes = ['target', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
        else:
            toxicity_classes = ['target']
        other_classes = ['funny', 'wow', 'sad', 'likes', 'disagree']
        protected_attr = ['female', 'male', 'transgender', 'other_gender', 'white', 'asian', 'black', 'latino',
                          'other_race_or_ethnicity', 'atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim',
                          'other_religion',
                          'heterosexual', 'bisexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation',
                          'intellectual_or_learning_disability', 'physical_disability', 'psychiatric_or_mental_illness',
                          'other_disability']

        # filter for samples with annotator count >= 3 (identity + target) and non-ambiguous labels for all classes
        ds_train = ds['train'].filter \
            (lambda example: (example['identity_annotator_count'] > 0 and example['toxicity_annotator_count'] > 0))
        ds_test = ds['test_public_leaderboard'].filter \
            (lambda example: (example['identity_annotator_count'] > 0 and example['toxicity_annotator_count'] > 0))
        for target in toxicity_classes:
            ds_train = ds_train.filter(lambda example: (example[target] < 0.34 or example[target] > 0.66))
            ds_test = ds_test.filter(lambda example: (example[target] < 0.34 or example[target] > 0.66))

        X_train = ds_train['comment_text']
        X_test = ds_test['comment_text']

        # create one-hot labels and identity labels (due to previous filtering labels are either < 0.33 or > 0.66)
        y_train = np.zeros((len(ds_train), len(toxicity_classes)))
        y_test = np.zeros((len(ds_test), len(toxicity_classes)))
        g_train = np.zeros((len(ds_train), len(protected_attr)))
        g_test = np.zeros((len(ds_test), len(protected_attr)))
        for j, target in enumerate(toxicity_classes):
            y_train[:, j] = (np.asarray(ds_train[target]) > 0.5).astype(float)
            y_test[:, j] = (np.asarray(ds_test[target]) > 0.5).astype(float)
        for j, target in enumerate(protected_attr):
            # need 2/3 majority for identity labels, otherwise assume identitiy not mentioned (bc we explicitly look at those with identity label 1)
            g_train[:, j] = (np.asarray(ds_train[target]) > 0.66).astype(float)
            g_test[:, j] = (np.asarray(ds_test[target]) > 0.66).astype(float)

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = protected_attr

        class_weights = compute_class_weights(y_train, toxicity_classes)
        _ = compute_class_weights(y_test, toxicity_classes)  # assert all classes represented in y_test

    elif dataset_name == "bios-supervised":
        with open(local_dir, 'rb') as handle:
            merged_dataset = pickle.load(handle)

        keys_to_copy = ['hard_text', 'profession', 'gender', 'raw', 'titles_supervised', 'review', 'valid', 'name']
        classes = ['architect', 'surgeon', 'dentist', 'teacher', 'psychologist', 'nurse', 'photographer', 'physician',
                   'attorney', 'journalist']

        # multi-label only reviewed+valid
        ds = filter_bios_dataset(merged_dataset, classes, keys_to_copy, False, True, True)

        X_train = ds['train']['hard_text']
        y_train = np.asarray(ds['train']['label'])
        g_train = ds['train']['gender']
        X_test = ds['test']['hard_text'] + ds['dev']['hard_text']
        y_test = np.asarray(ds['test']['label'] + ds['dev']['label'])
        g_test = ds['test']['gender'] + ds['dev']['gender']

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['male', 'female']

        class_weights = compute_class_weights(y_train, classes)
        _ = compute_class_weights(y_test, classes)  # assert all classes represented in y_test

    elif dataset_name == "bios-unsupervised":
        ds = load_dataset("LabHC/bias_in_bios")
        X_train = ds['train']['hard_text']
        y_train = ds['train']['profession']
        g_train = ds['train']['gender']
        X_test = ds['test']['hard_text'] + ds['dev']['hard_text']
        y_test = ds['test']['profession'] + ds['dev']['profession']
        g_test = ds['test']['gender'] + ds['dev']['gender']
        class_weights = None

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['male', 'female']

    elif dataset_name == "twitterAAE":
        # Blodgett et al. (ACL 2018)
        # using the huggingface version: https://huggingface.co/datasets/lighteval/TwitterAAE
        ds_aa = load_dataset('lighteval/TwitterAAE', 'aa')
        ds_white = load_dataset('lighteval/TwitterAAE', 'white')

        # test set only, no labels
        n_per_group = 50000
        X_train = []
        X_test = ds_aa['test']['tweet'] + ds_white['test']['tweet']
        y_train = []
        y_test = [-1 for i in range(2 * n_per_group)]
        g_train = []
        g_test = [0 for i in range(n_per_group)] + [1 for i in range(n_per_group)]
        class_weights = None

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['aa', 'white']

    elif dataset_name == 'crows_pairs':
        dataset = load_dataset(dataset_name, split='test')
        X_test, y_test, g_test, protected_groups = preprocess_crowspairs(dataset)
        X_train = []
        y_train = []
        g_train = []
        class_weights = None

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test

        protected_attributes['labels'] = protected_groups

    else:
        print("dataset %s not supported yet" % dataset_name)
        return [], [], [], []

    if type(y_test) == list and not type(y_test[0]) == list:  # single label
        n_classes = np.max(y_test) + 1
        multi_label = False
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    elif type(y_test) == np.ndarray:
        n_classes = y_test.shape[1]
        multi_label = True
    else:  # list of lists
        n_classes = len(y_test[0])
        multi_label = True
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attributes
