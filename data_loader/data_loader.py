from .stereoset import StereoSet
from .bios import BiosDataset
from .crowspairs import CrowSPairs
from .jigsaw import JigsawBias
from .twitter_aae import TwitterAAE
from .sbic import SBICDataset


def get_dataset(dataset_name, local_dir=None):

    if "jigsaw" in dataset_name:
        if dataset_name == "jigsaw-multi":
            dataset = JigsawBias(local_dir=local_dir, option='multi-class')
        else:
            dataset = JigsawBias(local_dir=local_dir, option='single-class')
    elif dataset_name == "bios-supervised":
        dataset = BiosDataset(local_dir=local_dir, option='supervised')
    elif dataset_name == "bios-unsupervised":
        dataset = BiosDataset(local_dir=local_dir, option='unsupervised')
    elif dataset_name == "twitterAAE":
        dataset = TwitterAAE()
    elif dataset_name == 'crows_pairs':
        dataset = CrowSPairs()
    elif dataset_name == 'stereoset':
        dataset = StereoSet(option='both')
    elif dataset_name == 'sbic':
        dataset = SBICDataset(local_dir=local_dir)
    # TODO Implicit Hate
    else:
        print("dataset %s not supported yet" % dataset_name)
        return None

    return dataset
