import difflib
import string
import numpy as np

PUNCTUATION = string.punctuation.replace('-', '')

GROUPS_BY_BIAS_TYPES = {"race-color": ["black", "white", "asian"],
                        "socioeconomic": ["poor", "rich"],
                        "gender": ["male", "female"],
                        "sexual-orientation": ["homosexual", "heterosexual", "bisexual"],
                        "religion": ["christian", "muslim", "jewish", "buddhist", "hindu"],
                        "age": ["old", "young"],
                        "physical-appearance": ["overweight", "underweight", "fit", "healthy", "tall", "short", "average_height", "good_looking", "bad_looking"],
                        "disability": ["physical disability, sickness", "mental_disability_illness", "able bodied, non-stigmatized sickness", "mentally fit, neurotypical", "physically fit, gifted", "mentally gifted"],
                        "nationality": ["citizen", "foreigner, immigrant, undocumented", "iraq", "canadian", "british", "mexican", "us american", "russian", "polish", "armenian", "italian", "nigerian", "greek", "vietnamese", "irish", "chinese", "indian", "japanese", "saudi arabian", "australian", "dutch", "mongolian", "korean", "syrian", "kenyan", "swedish", "colombian", "pakistani", "thai", "german", "norwegian"]}

TERMS_BY_GROUPS = {
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
}

GROUPS_TO_LABEL = ['black', 'white', 'asian', 'poor', 'rich', 'male', 'female', 'physical disability, sickness', 'mental_disability_illness', 'foreigner, immigrant, undocumented', 'canadian', 'us american', 'mexican', 'chinese', 'italian', 'british', 'russian', 'indian', 'homosexual', 'heterosexual', 'overweight', 'underweight', 'fit', 'tall', 'short', 'christian', 'muslim', 'jewish', 'old', 'young']


def get_group_label(modified_terms: list, bias_type: str):
    if not bias_type in GROUPS_BY_BIAS_TYPES.keys():
        return None, None
    assert len(modified_terms) > 0

    group_lbl = None
    terms_missing = {group: [] for group in GROUPS_BY_BIAS_TYPES[bias_type]}
    for group in GROUPS_BY_BIAS_TYPES[bias_type]:
        group_terms = TERMS_BY_GROUPS[group]
        for term in modified_terms:
            if not term in group_terms:
                terms_missing[group].append(term)
        if len(terms_missing[group]) == 0:
            group_lbl = group
            break

    missing = []
    for group in GROUPS_BY_BIAS_TYPES[bias_type]:
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
        if ' ' + term + ' ' in sent or sent[-len(term):] == term or sent[:len(term)] == term:
            return True

    return False


def preprocess_crowspairs(dataset):
    n_sent = len(dataset) * 2  # each sample includes two sentences
    n_groups = len(GROUPS_TO_LABEL)

    bias_types = dataset.info.features['bias_type'].names
    labels = dataset.info.features['stereo_antistereo'].names

    protected_groups = GROUPS_TO_LABEL
    X_test = []
    y_test = []  # [-1 for i in range(n_sent)]
    g_test = np.zeros((n_sent, n_groups))

    for sample in dataset:
        bias_type = bias_types[sample['bias_type']]
        mod1, mod2 = get_diff(sample['sent_more'], sample['sent_less'])
        sample['group_more'], sample['terms_missing_more'] = get_group_label(mod1, bias_type)
        sample['group_less'], sample['terms_missing_less'] = get_group_label(mod2, bias_type)
        is_valid = sample['group_more'] is not None and sample['group_less'] is not None and sample['group_more'] != \
                   sample['group_less']

        X_test.append(sample['sent_more'])
        X_test.append(sample['sent_less'])
        y_test.append(sample['stereo_antistereo'])
        y_test.append(1 - sample['stereo_antistereo'])

        idx_more = len(X_test) - 2
        idx_less = len(X_test) - 1
        for idx_group, group in enumerate(protected_groups):
            if group_mentioned_in_sentence(sample['sent_more'], TERMS_BY_GROUPS[group]):
                g_test[idx_more, idx_group] = 1
            if group_mentioned_in_sentence(sample['sent_less'], TERMS_BY_GROUPS[group]):
                g_test[idx_less, idx_group] = 1

    return X_test, y_test, g_test, protected_groups