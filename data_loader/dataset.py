import numpy as np
from collections.abc import Callable
from sklearn.utils import shuffle
import math
import itertools


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
    n_samples = y_train.shape[0]
    # relative weight of 1s per class (compared to 0s not other classes!)
    class_weights = np.asarray([((n_samples - samples_per_class[lbl]) / samples_per_class[lbl]) for lbl in
                                classes])

    for lbl in classes:  # need to verify after filtering!
        assert samples_per_class[lbl] > 0, "no samples for class %s" % lbl

    return class_weights


class CustomDataset:

    def __init__(self, local_dir: str = None, n_folds: int = 4, random_state: int = 42):
        self.name = None
        self.local_dir = local_dir
        self.is_shuffled = False

        self.data = {}
        self.data_preprocessed = {}
        self.labels = {}
        self.protected_groups = {}

        self.class_names = []
        self.group_names = []

        self.class_weights = {}
        self.group_weights = {}

        self.splits = []
        self.n_samples = {}  # per split
        self.multi_label = False
        self.unlabeled = False
        self.n_groups = 0
        self.n_classes = 0

        self.n_folds = n_folds
        self.n_per_fold = 0
        self.random_state = random_state

        self.ids_for_cv = []
        self.ids_per_fold = []
        self.data_folds = []
        self.data_prep_folds = []
        self.label_folds = []
        self.group_folds = []
        self.class_weight_folds = []
        self.group_weight_folds = []

    def prepare(self):
        self.splits = list(self.data.keys())
        self.check_data_valid()

        self.n_samples = {split: len(self.data[split]) for split in self.splits}

        for split in self.splits:
            self.data_preprocessed[split] = None

        # class and protected group labels are assumed to be numpy arrays, either
        # binary multi-label (onehot encoded) with shape (n_samples, n_classes)
        # or (multi-class) single-label with shape (n_samples, 1) and values in [0, n_classes-1]
        # protected groups are always onehot encoded (see check_data_valid())
        self.multi_label = self.labels[self.splits[0]].shape[1] > 1
        self.n_groups = self.protected_groups[self.splits[0]].shape[1]
        if self.multi_label:
            self.n_classes = self.labels[self.splits[0]].shape[1]
        else:
            self.n_classes = np.max(self.labels[self.splits[0]]) + 1

        assert len(self.group_names) == self.n_groups
        assert len(self.class_names) == self.n_classes

        # compute class and group weights
        for split in self.splits:
            if self.multi_label:
                self.class_weights[split] = compute_class_weights(self.labels[split], self.class_names)
            else:
                self.class_weights[split] = None
            self.group_weights[split] = compute_class_weights(self.protected_groups[split], self.group_names)

        # set up CV folds if no train/test split is available
        if len(self.splits) == 1:
            self.n_per_fold = math.ceil(self.n_samples[self.splits[0]] / self.n_folds)
            self.ids_for_cv = np.arange(self.n_samples[self.splits[0]])
            if not self.is_shuffled:
                self.ids_for_cv = shuffle(self.ids_for_cv, random_state=self.random_state)
            self.ids_per_fold = [self.ids_for_cv[i:i + self.n_per_fold]
                                 for i in range(0, len(self.ids_for_cv), self.n_per_fold)]
            self.data_folds = [[self.data[self.splits[0]][idx] for idx in self.ids_per_fold[i]]
                               for i in range(self.n_folds)]
            self.label_folds = [self.labels[self.splits[0]][self.ids_per_fold[i], :] for i in range(self.n_folds)]
            self.group_folds = [self.protected_groups[self.splits[0]][self.ids_per_fold[i], :]
                                for i in range(self.n_folds)]

            # class weights per fold
            for fold_id in range(self.n_folds):
                if self.multi_label:
                    self.class_weight_folds.append(compute_class_weights(self.label_folds[fold_id],
                                                                             self.class_names))
                self.group_weight_folds.append(compute_class_weights(self.group_folds[fold_id], self.group_names))

    def check_data_valid(self):
        # splits consistent in all dictionaries
        err_msg = "mismatch of specified splits in data, labels and protected groups"
        assert self.splits == list(self.labels.keys()) == list(self.protected_groups.keys()), err_msg
        for split in self.splits:
            err_msg = "size mismatch between data, labels and protected groups for split %s" % split
            assert len(self.data[split]) == len(self.labels[split]) == len(self.protected_groups[split]), err_msg

        # check label shape and range
        assert self.labels[self.splits[0]].ndim == 2, "expected a 2D array for class labels"
        assert self.protected_groups[self.splits[0]].ndim == 2, "expected a 2D array for protected group labels"

        # onehot encoded group labels
        for split in self.splits:
            assert is_onehot(self.protected_groups[split]), ("protected group labels are not onehot-encoded for "
                                                             "split %s") % split

    def load(self):
        pass

    def set_cv_fold_preprocessed_data(self):
        if len(self.splits) == 1:
            if isinstance(self.data_preprocessed, np.ndarray):
                self.data_prep_folds = [self.data_preprocessed[self.splits[0]][self.ids_per_fold[i], :]
                                        for i in range(self.n_folds)]
            else:  # list
                self.data_prep_folds = [np.asarray([self.data_preprocessed[self.splits[0]][idx]
                                                    for idx in self.ids_per_fold[i]])
                                        for i in range(self.n_folds)]

    def preprocess_data(self, preprocess_fct: Callable):
        for split in self.splits:
            self.data_preprocessed[split] = preprocess_fct(self.data[split])
        print("preprocessed data has type: ", type(self.data_preprocessed[self.splits[0]]))
        self.set_cv_fold_preprocessed_data()

    def set_preprocessed_data(self, data_prep: dict):
        for split in self.splits:
            assert split in data_prep.keys()
            assert len(self.data[split]) == len(data_prep[split])
            self.data_preprocessed[split] = data_prep[split]
        print("preprocessed data has type: ", type(self.data_preprocessed[self.splits[0]]))
        self.set_cv_fold_preprocessed_data()

    def get_split(self, split: str):
        assert split in self.splits, ("tried to access a split that doesn't exist for this dataset: "
                                      "%s" % split)
        return (self.data[split], self.data_preprocessed[split], self.labels[split], self.protected_groups[split],
                self.class_weights[split], self.group_weights[split])

    def get_cv_split(self, fold_id: int):
        assert self.n_folds is not None and self.n_folds > 0, "CV folds are not set up!"
        assert 0 <= fold_id < self.n_folds, "got invalid fold_id: %i" % fold_id

        preprocessing_done = self.data_preprocessed[self.splits[0]] is not None

        class_weights_train = None
        class_weights_test = None
        data_prep_train = None
        data_prep_test = None

        # test data
        if self.multi_label:
            class_weights_test = self.class_weight_folds[fold_id]
        if preprocessing_done:
            data_prep_test = self.data_prep_folds[fold_id]
        data = {'test': (self.data_folds[fold_id], data_prep_test, self.label_folds[fold_id],
                         self.group_folds[fold_id], class_weights_test, self.group_weight_folds[fold_id])}

        # train data
        data_train = list(
            itertools.chain.from_iterable([fold for i, fold in enumerate(self.data_folds) if i != fold_id]))

        if preprocessing_done:
            data_prep_train = np.vstack([fold for i, fold in enumerate(self.data_prep_folds) if i != fold_id])

        labels_train = np.vstack([fold for i, fold in enumerate(self.label_folds) if i != fold_id])
        groups_train = np.vstack([fold for i, fold in enumerate(self.group_folds) if i != fold_id])
        if self.multi_label:
            class_weights_train = np.mean([fold for i, fold in enumerate(self.class_weight_folds) if i != fold_id], axis=0)
        group_weights_train = np.mean([fold for i, fold in enumerate(self.group_weight_folds) if i != fold_id], axis=0)

        data['train'] = (data_train, data_prep_train, labels_train, groups_train, class_weights_train,
                         group_weights_train)
        return data


