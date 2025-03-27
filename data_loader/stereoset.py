from .dataset import CustomDataset
from .helper import simplify_text

import datasets
import numpy as np

terms_race = {'european/white': ['ukrainian', 'norway', 'spain', 'italy', 'britain', 'norweigan', 'crimean', 'russian'],
              'african/black': ['sierra leon', 'eriteria', 'somalia', 'african', 'eritrean', 'sierra leone', 'somalia', 'morocco', 'ethiopian', 'ethiopia', 'ghanaian', 'cameroon', 'cape verde'],
              'hispanic': ['ecuador', 'spain', 'columbian', 'hispanic'],
              'south east asian': ['indian', 'japanese', 'vietnam', 'bengali', 'bangladesh'],
              'middle east': ['lebanon', 'iranian', 'persian', 'iraq', 'yemen', 'syrian', 'arab', 'saudi arabia', 'jordan', 'syria', 'saudi arabian', 'afghanistan', 'afghanistansuch']
              }

terms_gender = {'male': ['he', 'his', 'himself', 'man', 'men', 'father', 'male', 'schoolboy', 'gentlemen', 'brother', 'daddy', 'father', 'grandfather', 'gentlement'],
                'female': ['she', 'her', 'herself', 'woman', 'women', 'mother', 'female', 'schoolgirl', 'sister', 'mommy', 'grandmother']
                }

terms_religion = {'christian': ['bible', 'catholics'],
                  'muslim': ['muslim', 'mosque', 'islam'],
                  'hindu': ['brahmin']
                  }

n_groups = 0
group2idx = {}
for group_name in list(terms_race.keys())+list(terms_gender.keys())+list(terms_religion.keys()):
    group2idx.update({group_name: n_groups})
    n_groups += 1


class StereoSet(CustomDataset):

    def __init__(self, local_dir: str = None, option: str = 'both'):
        super().__init__(local_dir)

        valid_options = ['intra', 'inter', 'both']
        err_str = (("Stereoset has two subsets: 'intersentence' and 'intrasentence'. Either one of these subsets can "
                    "be loaded or a merged version. Choose one of the following options for the option parameter: ")
                   + str(valid_options))
        assert option in valid_options, err_str
        self.option = option

        self.name = 'stereoset'
        self.group_names = list(group2idx.keys())

        print("load Stereoset with option %s" % self.option)
        self.load(local_dir)
        self.prepare()

    @staticmethod
    def process_dataset_version(version):
        assert version in ['intersentence', 'intrasentence']

        ds = datasets.load_dataset('McGill-NLP/stereoset', version)
        class_names = ds['validation'].info.features['sentences'].feature['gold_label'].names

        # every sample has 3 sentences with different labels
        # there is only the validation set
        n_sent = len(ds['validation']) * 3
        g_val = np.zeros((n_sent, n_groups))
        y_val = np.zeros((n_sent, 1))
        sent_val = []

        for i, sample in enumerate(ds['validation']):
            context = simplify_text(sample['context'])
            context_ = ' ' + context + ' '
            sentences = [simplify_text(sent) for sent in sample['sentences']['sentence']]
            assert len(sentences) == 3, "got n!=3 sentences: %s" % sentences
            labels = sample['sentences']['gold_label']

            groups = []
            for group_dict in [terms_race, terms_gender, terms_religion]:
                for group, terms in group_dict.items():
                    for term in terms:
                        if (' %s ' % term) in context_ or (' %ss ' % term) in context_:
                            groups.append(group)

            sent_idx = i * 3
            for j, sent in enumerate(sentences):
                if version == 'intersentence':
                    sent_val.append(context + '. ' + sent)
                else:
                    sent_val.append(sent)
                y_val[sent_idx + j] = labels[j]

            # set groups (each sample -> 3 sentences/labels)
            for group in groups:
                g_val[sent_idx:sent_idx + 2, group2idx[group]] = 1

        return sent_val, y_val, g_val, class_names

    def load(self, local_dir=None):
        if self.option == 'intra':
            print("Load only intrasentence samples")
            self.data['val'], self.labels['val'], self.protected_groups['val'], self.class_names = self.process_dataset_version('intrasentence')

        elif self.option == 'inter':
            print("Load only intersentence samples")
            self.data['val'], self.labels['val'], self.protected_groups['val'], self.class_names = self.process_dataset_version('intersentence')

        else:  # both
            print("Load inter- and intrasentence samples and merge them to one dataset")
            sent_val_inter, y_val_inter, g_val_inter, self.class_names = self.process_dataset_version('intersentence')
            sent_val_intra, y_val_intra, g_val_intra, _ = self.process_dataset_version('intrasentence')

            # note that the dataset is ordered and needs to be shuffled before training/ splitting!
            self.data['val'] = sent_val_inter + sent_val_intra
            self.labels['val'] = np.vstack([y_val_inter, y_val_intra])
            self.protected_groups['val'] = np.vstack([g_val_inter, g_val_intra])


