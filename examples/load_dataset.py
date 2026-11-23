import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader

prefix = '../../data/sbic_reannot/' # adapt data if stored elsewhere

local_paths = {
    'data': prefix+'review_full.csv',
    'label_map_orig': prefix+'freetext_label_lookup_original.csv',
    'label_hierarchy': prefix+'label_hierarchy.csv',
    'label_category': prefix+'label_categories.csv'
}

print("load subset with original labels: ")
sbic = data_loader.get_dataset('sbic-sorig', local_dir=local_paths)
texts, _, labels, groups, cw, gw = sbic.get_split('test')

print(texts[:10])
print(labels)
print(groups)
print(cw)
print(gw)
print()


print("load data from balanced annot study: ")
sbic = data_loader.get_dataset('sbic-sbalanced', local_dir=local_paths)
texts, _, labels, groups, cw, gw = sbic.get_split('test')

print(texts[:10])
print(labels)
print(groups)
print(cw)
print(gw)
print()

print("load data from black / aae annot study: ")

local_paths = {
    'data': prefix+'aae_annot.csv',
    'label_map_orig': prefix+'freetext_label_lookup_original.csv',
    'label_hierarchy': prefix+'label_hierarchy.csv',
    'label_category': prefix+'label_categories.csv'
}
# options: full sorig sbalanced sblack
sbic = data_loader.get_dataset('sbic-sblack', local_dir=local_paths)
texts, _, labels, groups, cw, gw = sbic.get_split('test')

print(texts[:10])
print(labels)
print(groups)
print(cw)
print(gw)


# full dataset
sbic = data_loader.get_dataset('sbic-full', local_dir=local_paths)