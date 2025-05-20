import datasets
import numpy as np

from .dataset import CustomDataset
from .helper import simplify_text


TERMS_BY_GROUPS = {
    "black": ["black", "black people", "blacks", "african", "africans", "african-american", "african american", "afro-american", "african-americans", "african americans", "an afro-american", "african american", "african americans", "colored", "dark-skinned"],
    "white": ["white", "white people", "whites", "caucasian", "caucasians", "white american", "caucasian american", "anglo-american", "white americans", "caucasian americans", "a caucasian american", "anglo americans", "anglo american", "white", "light-skinned"],
    "asian": ["asian", "asian people", "asians", "asian", "asians", "asian-american", "asian american", "asian-american", "asian-americans", "asian americans", "an asian-american", "asian american", "asian americans", "asian", "asian",
              "vietnam", "vietnamese", "a vietnamese", "china", "chinese", "a chinese", "india", "indian", "indians", "an indian", "japan", "japanese", "a japanese", "mongolia", "mongolian", "mongolians", "a mongolian", "korea", "korean",
              "koreans", "a korean", "pakistan", "pakistanti", "a pakistanti", "thailand", "thai", "a thai", "russia", "russian", "russians", "a russian", "armenia", "armenian", "armenians", "an armenian"],

    "native american": ["native", "native american", "native americans"],
    "hispanic": ["hispanic", "hispanics"],
    "latino": ["latino", "latinos", "mexico", "mexican", "mexicans", "a mexican", "colombia", "colombian", "colombians", "a colombian"],
    "african": ["nigeria", "nigerian", "nigerians", "a nigerian", "kenya", "kenyan", "kenyans", "a kenyan"],
    "middle eastern": ["saudi arabia", "saudi arabian", "saudi arabians", "a saudi arabian", "syria", "syrian", "syrians", "a syrian"],
    "european": ["great britian", "british", "english", "welsh", "scottish", "a brit", "an englishman", "a welsh", "a scotsman", "poland", "polish", "a polish", "italy", "italian", "italians", "an italian", "greece", "greek", "greeks",
                 "a greek", "ireland", "irish", "irishman", "an irishman", "germany", "german", "germans", "a german", "norway", "norwegian", "norwegians", "a norwegian", "netherlands", "dutch", "dutchman", "a dutchman", "sweden", "swede",
                 "swedish", "swedes", "a swede", "russia", "russian", "russians", "a russian"],
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
    "minors": ["teenager", "children", "child"],

    "overweight": ["fat", "overweight", "obese", "morbidly obese", "an obese", "heavy", "fatties"],
    "underweight": ["very thin", "thin", "skinny", "slim", "light", "underweight"],
    "fit": ["buff", "muscular", "strong", "fit", "sturdy", "in shape", "bulging muscles"],
    "healthy": ["of normal weight", "healthy"],
    "tall": ["tall", "tallest", "large"],
    "short": ["short", "shortest", "little person", "tiny", "small"],
    "average_height": ["of average height"],
    "good_looking": ["beautiful", "cute", "handsome", "attractive"],
    "bad_looking": ["ugly", "hunchback", "disfigured"],

    "physical disability": ["wheelchair", "sickly", "disabled", "handicapped", "no legs", "hospital", "crippled",
                            "a crippled", "paraplegic", "blind eye", "blind", "glasses"],
    "physical illness": ["wheelchair", "sickly", "hospital", "crippled", "a crippled", "aids"],
    "mental disability": ["dyslexia", "retard", "retarded", "retards", "autistic", "down syndrome"],
    "mental illness": ["mentally ill", "demented", "alcoholic", "drug addict"],

    # nationalities, some not-protected groups (as counterparts to disability)
    "able bodied, non-stigmatized sickness": ["healthy", "able-bodied", "able bodied", "all limbs", "able", "nondisabled", "of sound mind", "healthy man", "the cold"],
    "mentally fit, neurotypical": ["normal", "average iq", "mentally fit"],
    "physically fit, gifted": ["perfectly healthy", "athletic"],
    "mentally gifted": ["smart", "genius", "math prodigy", "clever", "superb", "gifted"],

    "citizen": ["a citizen", "citizen"],
    "foreigner, immigrant, undocumented": ["foreigner", "immigrants", "refugee", "undocumented", "illegals"],

    "canadian": ["canada", "canadian", "canadians", "a canadian"],
    "mexican": ["mexico", "mexican", "mexicans", "a mexican"],
    "us american": ["american", "usa", "americans", "an american"],

    "colombian": ["colombia", "colombian", "colombians", "a colombian"],

    "british": ["great britian", "british", "english", "welsh", "scottish", "a brit", "an englishman", "a welsh", "a scotsman"],
    "polish": ["poland", "polish", "a polish"],
    "italian": ["italy", "italian", "italians", "an italian"],
    "greek": ["greece", "greek", "greeks", "a greek"],
    "irish": ["ireland", "irish", "irishman", "an irishman"],
    "german": ["germany", "german", "germans", "a german"],
    "norwegian": ["norway", "norwegian", "norwegians", "a norwegian"],
    "dutch": ["netherlands", "dutch", "dutchman", "a dutchman"],
    "swedish": ["sweden", "swede", "swedish", "swedes", "a swede"],

    "nigerian": ["nigeria", "nigerian", "nigerians", "a nigerian"],
    "kenyan": ["kenya", "kenyan", "kenyans", "a kenyan"],

    "australian": ["australia", "australian", "australians", "an australian"],

    "russian": ["russia", "russian", "russians", "a russian"],
    "armenian": ["armenia", "armenian", "armenians", "an armenian"],

    "saudi arabian": ["saudi arabia", "saudi arabian", "saudi arabians", "a saudi arabian"],
    "syrian": ["syria", "syrian", "syrians", "a syrian"],

    "vietnamese": ["vietnam", "vietnamese", "a vietnamese"],
    "chinese": ["china", "chinese", "a chinese"],
    "indian": ["india", "indian", "indians", "an indian"],
    "japanese": ["japan", "japanese", "a japanese"],
    "mongolian": ["mongolia", "mongolian", "mongolians", "a mongolian"],
    "korean": ["korea", "korean", "koreans", "a korean"],
    "pakistani": ["pakistan", "pakistanti", "a pakistanti"],
    "thai": ["thailand", "thai", "a thai"],
}

GROUPS_TO_LABEL = ['black', 'white', 'asian',
                   'poor', 'rich',
                   'male', 'female',
                   'physical disability', 'mental disability', 'physical illness', 'mental illness',
                   'foreigner, immigrant, undocumented', 'canadian', 'us american', 'mexican', 'chinese', 'italian', 'british', 'russian', 'indian',
                   'homosexual', 'heterosexual', 'bisexual',
                   'overweight', 'underweight', 'fit', 'tall', 'short', 'old', 'young',
                   'christian', 'muslim', 'jewish']


def group_mentioned_in_sentence(sentence, group_terms):
    sent = simplify_text(sentence)
    for term in group_terms:
        if ' ' + term + ' ' in sent or sent[-len(term):] == term or sent[:len(term)] == term:
            return True

    return False


class CrowSPairs(CustomDataset):

    def __init__(self, local_dir: str = None):
        super().__init__(local_dir)

        self.name = 'crowspairs'
        self.group_names = GROUPS_TO_LABEL

        print("load crowspairs")
        self.load(local_dir)
        self.prepare()

    def load(self, local_dir=None):
        dataset = datasets.load_dataset('crows_pairs', split='test', trust_remote_code=True)
        n_sent = len(dataset) * 2  # each sample includes two sentences

        bias_types = dataset.info.features['bias_type'].names
        self.class_names = dataset.info.features['stereo_antistereo'].names
        n_groups = len(self.group_names)

        self.data['test'] = []
        self.labels['test'] = []
        self.protected_groups['test'] = np.zeros((n_sent, n_groups))

        for sample in dataset:
            self.data['test'].append(sample['sent_more'])
            self.data['test'].append(sample['sent_less'])
            self.labels['test'].append(sample['stereo_antistereo'])
            self.labels['test'].append(1 - sample['stereo_antistereo'])

            idx_more = len(self.data['test']) - 2
            idx_less = len(self.data['test']) - 1
            for idx_group, group in enumerate(self.group_names):
                if group_mentioned_in_sentence(sample['sent_more'], TERMS_BY_GROUPS[group]):
                    self.protected_groups['test'][idx_more, idx_group] = 1
                if group_mentioned_in_sentence(sample['sent_less'], TERMS_BY_GROUPS[group]):
                    self.protected_groups['test'][idx_less, idx_group] = 1

        self.labels['test'] = np.asarray(self.labels['test']).reshape(-1, 1)

