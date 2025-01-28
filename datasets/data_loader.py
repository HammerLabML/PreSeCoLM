import numpy as np
import pickle

from datasets import load_dataset, Dataset, DatasetDict
import difflib
import string

PUNCTUATION = string.punctuation.replace('-', '')

groups_by_bias_types = {"race-color": ["black", "white", "asian"], 
                        "socioeconomic": ["poor", "rich"], 
                        "gender": ["male", "female"], 
                        "sexual-orientation": ["homosexual", "heterosexual", "bisexual"], 
                        "religion": ["christian", "muslim", "jewish", "buddhist", "hindu"],
                        "age": ["old", "young"],
                        "physical-appearance": ["overweight", "underweight", "fit", "healthy", "tall", "short", "average_height", "good_looking", "bad_looking"],
                        "disability": ["physical disability, sickness", "mental_disability_illness", "able bodied, non-stigmatized sickness", "mentally fit, neurotypical", "physically fit, gifted", "mentally gifted"],
                        "nationality": ["citizen", "foreigner, immigrant, undocumented", "iraq", "canadian", "british", "mexican", "us american", "russian", "polish", "armenian", "italian", "nigerian", "greek", "vietnamese", "irish", "chinese", "indian", "japanese", "saudi arabian", "australian", "dutch", "mongolian", "korean", "syrian", "kenyan", "swedish", "colombian", "pakistani", "thai", "german", "norwegian"]}
     

terms_by_groups = {
         "black": ["black", "black people", "blacks", "african", "africans", "african-american", "african american", "afro-american", "african-americans", "african americans", "an afro-american", "african american", "african americans", "colored", "dark-skinned"], 
         "white": ["white", "white people", "whites", "caucasian", "caucasians", "white american", "caucasian american", "anglo-american", "white americans", "caucasian americans", "a caucasian american", "anglo americans", "anglo american", "white", "light-skinned"], 
         "asian": ["asian", "asian people", "asians", "asian", "asians", "asian-american", "asian american", "asian-american", "asian-americans", "asian americans", "an asian-american", "asian american", "asian americans", "asian", "asian"], 
         
         "native american": ["native", "native american", "native americans"], "hispanic": ["latino", "latinos", "hispanic", "hispanics"], 
         # other ethnicities
         
         "poor": ["poor", "ghetto", "low income", "poverty stricken", "poverty", "low income", "trailer park", "welfare money", "on welfare", "homeless"], 
         "rich": ["rich", "suburb", "high income", "over wealthy", "wealth", "wealthy", "high income", "high class", "gated community", "yacht", "billionaires"], 
         
         "male": ["he", "his", "his", "him", "himself", "man", "men", "dad", "father", "son", "grandson", "sons", "brother", "brothers", "grandfather", "grandpa", "uncle", "mr", "husband", "boy", "boys", "male", "masculine", "sir", "sir"], 
         "female": ["she", "her", "hers", "her", "herself", "woman", "women", "mom", "mother", "daughter", "granddaughter", "daughters", "sister", "sisters", "grandmother", "grandma", "aunt", "mrs", "wife", "girl", "girls", "female", "feminine", "maam", "madam"], 
         "trans": ["trans", "transgender", "transgender female", "transgender male"], 
         "cis": ["cis", "cisgender", "cisfemale", "cismale"], 
         
         "homosexual": ["homo", "gay", "lesbian", "homosexual"], 
         "heterosexual": ["hetero", "straight", "straight", "heterosexual"], 
         "bisexual": ["bi", "bi", "bi", "bisexual"], 
         
         "christian": ["christian", "christian", "christians", "church", "chapel", "bible", "bible", "christianity", "god", "a christian", "christmas", "easter", "catholic", "lutheran"], 
         "muslim": ["muslim", "muslim", "muslims", "mosque", "masjid", "quran", "quran", "islam", "allah", "a muslim", "ramadan", "eid al-adha", "muslim", "muslim"], 
         "jewish": ["jewish", "jew", "jews", "synagogue", "synagog", "tanakh", "hebrew bible", "judaism", "yahweh", "a jew", "hanukkah", "tisha bav", "jewish", "jewish"], 
         "buddhist": ["buddhist", "buddhist", "buddhists", "vihara", "buddhist temple", "buddhist canons", "buddhist canons", "buddhism", "buddha", "a buddhist", "vesak", "parinirvana day", "buddhist", "buddhist"], 
         "hindu": ["hindu", "hindu", "hindus", "mandir", "hindu temple", "vedas", "upanishads", "hinduism", "hindu deities", "a hindu", "diwali", "makar sankranti", "hindu", "hindu"], 
         
         "old": ["old", "older", "an old", "oldest", "elderly", "senior", "adults", "adult"], 
         "young": ["young", "younger", "a young", "youngest", "teenager", "children", "child", "junior"],
     
         "overweight": ["fat", "overweight", "obese", "morbidly obese", "an obese", "heavy", "fatties"],
         "underweight": ["very thin", "thin", "skinny", "slim", "light", "underweight"],
         "fit": ["buff", "muscular", "strong", "fit", "sturdy", "in shape", "bulging muscles"],
         "healthy": ["of normal weight", "healthy"],
         "tall": ["tall", "tallest", "large"],
         "short": ["short", "shortest", "little person", "tiny", "small"],
         "average_height": ["of average height"],
         "good_looking": ["beautiful", "cute", "handsome", "attractive"],
         "bad_looking": ["ugly", "hunchback", "disfigured"],

    # TODO: reconsider these categories
        "physical disability, sickness": ["wheelchair", "sickly", "disabled", "handicapped", "no legs", "hospital", "crippled", "a crippled", "paraplegic", "blind eye", "blind", "glasses", "aids"],
        "mental_disability_illness": ["dyslexia", "retard", "retarded", "retards", "mentally ill", "autistic", "down syndrome", "demented", "alcoholic", "drug addict"],
        "able bodied, non-stigmatized sickness": ["healthy", "able-bodied", "able bodied", "all limbs", "able", "nondisabled", "of sound mind", "healthy man", "the cold"],
        "mentally fit, neurotypical": ["normal", "average iq", "mentally fit"],
        "physically fit, gifted": ["perfectly healthy", "athletic"],
        "mentally gifted": ["smart", "genius", "math prodigy", "clever", "superb", "gifted"],
    
        "citizen": ["a citizen", "citizen"],
        "foreigner, immigrant, undocumented": ["foreigner", "immigrants", "refugee", "undocumented", "illegals"],
        "iraq": ["iraq", "iraqi", "an iraqui"],
        "canadian": ["canada", "canadian", "canadians", "a canadian"],
        "british": ["great britian", "british", "english", "welsh", "scottish", "a brit", "an englishman", "a welsh", "a scotsman"],
        "mexican": ["mexico", "mexican", "mexicans", "a mexican"],
        "us american": ["american", "usa", "americans", "an american"],
        "russian": ["russia", "russian", "russians", "a russian"],
        "polish": ["poland", "polish", "a polish"],
        "armenian": ["armenia", "armenian", "armenians", "an armenian"],
        "italian": ["italy", "italian", "italians", "an italian"],
        "nigerian": ["nigeria", "nigerian", "nigerians", "a nigerian"],
        "greek": ["greece", "greek", "greeks", "a greek"],
        "vietnamese": ["vietnam", "vietnamese", "a vietnamese"],
        "irish": ["ireland", "irish", "irishman", "an irishman"],
        "chinese": ["china", "chinese", "a chinese"],
        "indian": ["india", "indian", "indians", "an indian"],
        "japanese": ["japan", "japanese", "a japanese"],
        "saudi arabian": ["saudi arabia", "saudi arabian", "saudi arabians", "a saudi arabian"],
        "australian": ["australia", "australian", "australians", "an australian"],
        "dutch": ["netherlands", "dutch", "dutchman", "a dutchman"], 
        "mongolian": ["mongolia", "mongolian", "mongolians", "a mongolian"],
        "korean": ["korea", "korean", "koreans", "a korean"],
        "syrian": ["syria", "syrian", "syrians", "a syrian"],
        "kenyan": ["kenya", "kenyan", "kenyans", "a kenyan"],
        "swedish": ["sweden", "swede", "swedish", "swedes", "a swede"],
        "colombian": ["colombia", "colombian", "colombians", "a colombian"],
        "pakistani": ["pakistan", "pakistanti", "a pakistanti"],
        "thai": ["thailand", "thai", "a thai"],
        "german": ["germany", "german", "germans", "a german"],
        "norwegian": ["norway", "norwegian", "norwegians", "a norwegian"]
        # roma/ gypsy?
     }

crowspairs_groups_to_label = ['black', 'white', 'asian', 'poor', 'rich', 'male', 'female', 'physical disability, sickness', 'mental_disability_illness', 'foreigner, immigrant, undocumented', 'canadian', 'us american', 'mexican', 'chinese', 'italian', 'british', 'russian', 'indian', 'homosexual', 'heterosexual', 'overweight', 'underweight', 'fit', 'tall', 'short', 'christian', 'muslim', 'jewish', 'old', 'young']

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
        maxv = max(1,int(np.max(y)))
    onehot = np.zeros((len(y), 1+maxv-minv))
    onehot[np.arange(len(y)), y] = 1
    return onehot.astype('float')

# filter the dataset: from MeasuringFairnessWithBiasedData (TODO: github link)
def filter_bios_dataset(dataset: dict, classes: list, keys_to_copy: list, single_label=True, review_only=True, valid_only=True):
    splits = dataset.keys()
    split_dict = {}
    filtered_dataset = {split: [] for split in splits}
    for split in splits:
        for elem in dataset[split]:
            if valid_only and elem['valid'] != 1:
                continue
            if review_only and elem['review'] != 1:
                continue
            sel_titles = [title for title in elem['titles_supervised'] if title in classes]
            if single_label and len(sel_titles) > 1:
                continue
            if len(sel_titles) == 0:
                continue

            new_entry = {k: elem[k] for k in keys_to_copy}
            if single_label:
                label = classes.index(sel_titles[0])
            else:  # multi-label / one-hot encoded
                label = np.zeros(len(classes))
                for title in sel_titles:
                    label[classes.index(title)] = 1
            new_entry.update({'label': label})
            filtered_dataset[split].append(new_entry)
        print(len(filtered_dataset[split]))

        cur_split = {k: [elem[k] for elem in filtered_dataset[split]] for k in filtered_dataset[split][0].keys()}
        split_dict[split] = Dataset.from_dict(cur_split, split=split)
    return DatasetDict(split_dict)

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


def get_group_label(modified_terms: list, bias_type: str, groups_by_bias_types: dict, terms_by_groups: dict):
    if not bias_type in groups_by_bias_types.keys():
        return None, None
    assert len(modified_terms) > 0

    group_lbl = None
    terms_missing = {group: [] for group in groups_by_bias_types[bias_type]}
    for group in groups_by_bias_types[bias_type]:
        group_terms = terms_by_groups[group]
        for term in modified_terms:
            if not term in group_terms:
                terms_missing[group].append(term)
        if len(terms_missing[group]) == 0:
            group_lbl = group
            break

    missing = []
    for group in groups_by_bias_types[bias_type]:
        missing += terms_missing[group]

    return group_lbl, list(set(missing))


def simplify_text(text: str):
    return text.strip().lower().translate(str.maketrans('', '', PUNCTUATION))


def get_diff(seq1, seq2):
    seq1 = seq1.split(' ')
    seq2 = seq2.split(' ')
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    modified1 = []
    modified2 = []
    for op in matcher.get_opcodes():
        if not op[0] == 'equal':
            mod1 = ""
            mod2 = ""
            for x in range(op[1], op[2]):
                mod1 += ' ' + seq1[x]
            for x in range(op[3], op[4]):
                mod2 += ' ' + seq2[x]
            modified1.append(simplify_text(mod1))
            modified2.append(simplify_text(mod2))

    return modified1, modified2

def group_mentioned_in_sentence(sentence, group_terms):
    sent = simplify_text(sentence)
    for term in group_terms:
        if ' '+term+' ' in sent or sent[-len(term):] == term or sent[:len(term)] == term:
            return True

    return False

def preprocess_crowspairs(dataset, groups_by_bias_type: dict, group_terms: dict):
    n_sent = len(dataset)*2 # each sample includes two sentences
    n_groups = len(crowspairs_groups_to_label)
    
    bias_types = dataset.info.features['bias_type'].names
    labels = dataset.info.features['stereo_antistereo'].names

    protected_groups = crowspairs_groups_to_label
    X_test = []
    y_test = []#[-1 for i in range(n_sent)]
    g_test = np.zeros((n_sent,n_groups))

    for sample in dataset:
        bias_type = bias_types[sample['bias_type']]
        mod1, mod2 = get_diff(sample['sent_more'], sample['sent_less'])
        sample['group_more'], sample['terms_missing_more'] = get_group_label(mod1, bias_type, groups_by_bias_type, group_terms)
        sample['group_less'], sample['terms_missing_less'] = get_group_label(mod2, bias_type, groups_by_bias_type, group_terms)
        is_valid = sample['group_more'] is not None and sample['group_less'] is not None and sample['group_more']  != sample['group_less']

        X_test.append(sample['sent_more'])
        X_test.append(sample['sent_less'])
        y_test.append(sample['stereo_antistereo'])
        y_test.append(1-sample['stereo_antistereo'])

        idx_more = len(X_test)-2
        idx_less = len(X_test)-1
        for idx_group, group in enumerate(protected_groups):
            if group_mentioned_in_sentence(sample['sent_more'], group_terms[group]):
                g_test[idx_more,idx_group] = 1
            if group_mentioned_in_sentence(sample['sent_less'], group_terms[group]):
                g_test[idx_less,idx_group] = 1
        
    return X_test, y_test, g_test, protected_groups

def get_dataset(dataset_name, local_dir=None):
    protected_attributes = {'train': None, 'test': None, 'labels': []}
    
    # GLUE
    if dataset_name in ["sst2"]: # TODO
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
        protected_attr = ['female', 'male', 'transgender', 'other_gender', 'white', 'asian', 'black', 'latino', 'other_race_or_ethnicity', 'atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim', 'other_religion', 
                          'heterosexual', 'bisexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation', 'intellectual_or_learning_disability', 'physical_disability', 'psychiatric_or_mental_illness', 'other_disability']

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
            y_train[: ,j] = (np.asarray(ds_train[target]) > 0.5).astype(float)
            y_test[: ,j] = (np.asarray(ds_test[target]) > 0.5).astype(float)
        for j, target in enumerate(protected_attr):
            # need 2/3 majority for identity labels, otherwise assume identitiy not mentioned (bc we explicitly look at those with identity label 1)
            g_train[: ,j] = (np.asarray(ds_train[target]) > 0.66).astype(float)
            g_test[: ,j] = (np.asarray(ds_test[target]) > 0.66).astype(float)

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = protected_attr

        class_weights = compute_class_weights(y_train, toxicity_classes)
        _ = compute_class_weights(y_test, toxicity_classes) # assert all classes represented in y_test

    elif dataset_name=="bios-supervised":
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
        X_test = ds['test']['hard_text' ] +ds['dev']['hard_text']
        y_test = np.asarray(ds['test']['label' ] +ds['dev']['label'])
        g_test = ds['test']['gender']+ds['dev']['gender']

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['male','female']

        class_weights = compute_class_weights(y_train, classes)
        _ = compute_class_weights(y_test, classes) # assert all classes represented in y_test

    elif dataset_name=="bios-unsupervised":
        ds = load_dataset("LabHC/bias_in_bios")
        X_train = ds['train']['hard_text']
        y_train = ds['train']['profession']
        g_train = ds['train']['gender']
        X_test = ds['test']['hard_text'] + ds['dev']['hard_text']
        y_test = ds['test']['profession'] + ds['dev']['profession']
        g_test = ds['test']['gender']+ds['dev']['gender']
        class_weights = None

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['male','female']

    elif dataset_name=="twitterAAE":
        # Blodgett et al. (ACL 2018)
        # using the huggingface version: https://huggingface.co/datasets/lighteval/TwitterAAE
        ds_aa = load_dataset('lighteval/TwitterAAE', 'aa')
        ds_white = load_dataset('lighteval/TwitterAAE', 'white')

        # test set only, no labels
        n_per_group = 50000
        X_train = []
        X_test = ds_aa['test']['tweet'] + ds_white['test']['tweet']
        y_train = []
        y_test = [-1 for i in range(2*n_per_group)]
        g_train = []
        g_test = [0 for i in range(n_per_group)]+[1 for i in range(n_per_group)]
        class_weights = None

        protected_attributes['train'] = g_train
        protected_attributes['test'] = g_test
        protected_attributes['labels'] = ['aa','white']

    elif dataset_name=='crows_pairs':
        dataset = load_dataset(dataset_name, split='test')
        X_test, y_test, g_test, protected_groups = preprocess_crowspairs(dataset, groups_by_bias_types, terms_by_groups)
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

    if type(y_test) == list and not type(y_test[0]) == list: # single label
        n_classes = np.max(y_test)+1
        multi_label = False
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    elif type(y_test) == np.ndarray:
        n_classes = y_test.shape[1]
        multi_label = True
    else: # list of lists
        n_classes = len(y_test[0])
        multi_label = True
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test, n_classes, multi_label, class_weights, protected_attributes
