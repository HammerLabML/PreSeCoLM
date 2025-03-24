import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import BiosDataset, JigsawBias, CustomDataset, CrowSPairs, StereoSet, TwitterAAE

# todo test function including
# labels all numpy arrays
# consistent splits and n samples
# load each split/ cv splits
# preprocess data with embedding
# check label/ group lbl dim and list of classes/ groups consitent

# todo tests for each dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_instance = self.create_dataset_instance()

    def create_dataset_instance(self):
        raise NotImplementedError("Subclass must implement this")

    def test_labels(self):
        if self.dataset_instance.unlabeled:
            return
        n_classes = len(self.dataset_instance.class_names)
        self.assertTrue(n_classes > 0)
        for split in self.dataset_instance.splits:
            self.assertIsInstance(self.dataset_instance.labels[split], np.ndarray)
            if self.dataset_instance.multi_label:
                lbl_shape = self.dataset_instance.labels[split].shape[1]
                self.assertEqual(lbl_shape, n_classes, "shape[1]=%i, n_groups=%i" % (lbl_shape, n_classes))
            else:
                max_lbl = np.max(self.dataset_instance.labels[split])+1
                self.assertEqual(max_lbl, n_classes, "max_lbl=%i, n_groups=%i" % (max_lbl, n_classes))

    def test_group_labels(self):
        n_groups = len(self.dataset_instance.group_names)
        self.assertTrue(n_groups > 0)
        for split in self.dataset_instance.splits:
            self.assertIsInstance(self.dataset_instance.protected_groups[split], np.ndarray)
            self.assertEqual(self.dataset_instance.protected_groups[split].shape[1], n_groups, "shape[1]=%i, n_groups=%i" % (self.dataset_instance.protected_groups[split].shape[1], n_groups))

    def test_split_samples(self):
        n_splits = len(self.dataset_instance.splits)
        self.assertEqual(len(self.dataset_instance.data.keys()), n_splits)
        self.assertEqual(len(self.dataset_instance.labels.keys()), n_splits)
        self.assertEqual(len(self.dataset_instance.protected_groups.keys()), n_splits)

        for split in self.dataset_instance.splits:
            n_samples = len(self.dataset_instance.data[split])
            self.assertEqual(len(self.dataset_instance.labels[split]), n_samples)
            self.assertEqual(len(self.dataset_instance.protected_groups[split]), n_samples)

    def test_load_split(self):
        for split in self.dataset_instance.splits:
            data, data_pred, lbl, group_lbl, cw, gw = self.dataset_instance.get_split(split)
        if len(self.dataset_instance.splits) == 1:
            for fold_id in range(self.dataset_instance.n_folds):
                data_dict = self.dataset_instance.get_cv_split(fold_id)
                self.assertTrue('train' in data_dict.keys())
                self.assertTrue('test' in data_dict.keys())

    def test_preprocessing(self):
        # apply preprocessing to data
        pass


class TestBios(TestDataset):
    def create_dataset_instance(self):
        return BiosDataset(local_dir=None, option='unsupervised')


class TestStereoSet(TestDataset):
    def create_dataset_instance(self):
        return StereoSet(local_dir=None, option='both')


class TestTwitterAAE(TestDataset):
    def create_dataset_instance(self):
        return TwitterAAE(local_dir=None)


class TestCrowSPairs(TestDataset):
    def create_dataset_instance(self):
        return CrowSPairs()

# TODO: bios supervised and jigsaw with local_dir specified


if __name__ == '__main__':
    unittest.main()
