from .dataset import CustomDataset, label2onehot, is_onehot
from datasets import load_dataset
import numpy as np


class JigsawBias(CustomDataset):

    def __init__(self, local_dir: str = None, option: str = 'single-class'):
        super().__init__(local_dir)

        valid_options = ['single-class', 'multi-class']
        err_str = (("JigsawBias can be loaded as 'single-class' with a binary toxicity label or as 'multi-class' "
                    "as multi-class multi-label with all available labels (e.g. obscene, identity_attack, ...). "
                    "Choose one of the following options for the option parameter: ")
                   + str(valid_options))
        assert option in valid_options, err_str
        self.option = option

        self.name = 'jigsaw'
        self.group_names = ['female', 'male', 'transgender',
                            'white', 'asian', 'black', 'latino',
                            'atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim',
                            'heterosexual', 'bisexual', 'homosexual_gay_or_lesbian',
                            'intellectual_or_learning_disability', 'physical_disability',
                            'psychiatric_or_mental_illness']
        # removed the 'other*' labels, because there were no samples after filtering
        # 'other_gender', 'other_disability', 'other_sexual_orientation', 'other_religion', 'other_race_or_ethnicity'

        print("load JigsawBias with option: %s" % self.option)
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None):
        ds = load_dataset("jigsaw_unintended_bias", data_dir=local_dir, trust_remote_code=True)

        if self.option == "multi-class":
            self.class_names = ['target', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
            # other_classes = ['funny', 'wow', 'sad', 'likes', 'disagree']
        else:
            self.class_names = ['target']

        # filter for samples with annotator count >= 3 (identity + target) and non-ambiguous labels for all classes
        ds_train = (ds['train'].filter
                    (lambda example: (
                                example['identity_annotator_count'] > 0 and example['toxicity_annotator_count'] > 0)))
        ds_dev = (ds['test_public_leaderboard'].filter
                  (lambda example: (example['identity_annotator_count'] > 0 and example[
                      'toxicity_annotator_count'] > 0)))
        ds_test = (ds['test_private_leaderboard'].filter
                   (lambda example: (example['identity_annotator_count'] > 0 and example[
                       'toxicity_annotator_count'] > 0)))
        for target in self.class_names:
            ds_train = ds_train.filter(lambda example: (example[target] < 0.34 or example[target] > 0.66))
            ds_dev = ds_dev.filter(lambda example: (example[target] < 0.34 or example[target] > 0.66))
            ds_test = ds_test.filter(lambda example: (example[target] < 0.34 or example[target] > 0.66))

        self.data['train'] = ds_train['comment_text']
        self.data['dev'] = ds_dev['comment_text']
        self.data['test'] = ds_test['comment_text']

        # create one-hot labels and identity labels (due to previous filtering labels are either < 0.33 or > 0.66)
        self.protected_groups['train'] = np.zeros((len(ds_train), len(self.group_names)))
        self.protected_groups['dev'] = np.zeros((len(ds_dev), len(self.group_names)))
        self.protected_groups['test'] = np.zeros((len(ds_test), len(self.group_names)))

        if len(self.class_names) == 1:
            target = self.class_names[0]
            self.labels['train'] = (np.asarray(ds_train[target]) > 0.5).astype(float)
            self.labels['dev'] = (np.asarray(ds_dev[target]) > 0.5).astype(float)
            self.labels['test'] = (np.asarray(ds_test[target]) > 0.5).astype(float)

            # convert labels to onehot for BCE (which allows class weights)
            self.labels['train'] = label2onehot(self.labels['train'])
            self.labels['dev'] = label2onehot(self.labels['dev'])
            self.labels['test'] = label2onehot(self.labels['test'])
            self.class_names = ['not toxic', 'toxic']  # assumes this is the only 1class scenario
        else:
            self.labels['train'] = np.zeros((len(ds_train), len(self.class_names)))
            self.labels['dev'] = np.zeros((len(ds_dev), len(self.class_names)))
            self.labels['test'] = np.zeros((len(ds_test), len(self.class_names)))
            for j, target in enumerate(self.class_names):
                self.labels['train'][:, j] = (np.asarray(ds_train[target]) > 0.5).astype(float)
                self.labels['dev'][:, j] = (np.asarray(ds_dev[target]) > 0.5).astype(float)
                self.labels['test'][:, j] = (np.asarray(ds_test[target]) > 0.5).astype(float)

        for j, target in enumerate(self.group_names):
            # need 2/3 majority for identity labels, otherwise assume identity not mentioned
            # (bc we explicitly look at those with identity label 1)
            self.protected_groups['train'][:, j] = (np.asarray(ds_train[target]) > 0.66).astype(float)
            self.protected_groups['dev'][:, j] = (np.asarray(ds_dev[target]) > 0.66).astype(float)
            self.protected_groups['test'][:, j] = (np.asarray(ds_test[target]) > 0.66).astype(float)


