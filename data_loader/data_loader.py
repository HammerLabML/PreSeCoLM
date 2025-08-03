import pandas as pd
import numpy as np
from .dataset import CustomDataset


class ImplicitHateDataset(CustomDataset):
    def __init__(self, local_dir: str = None):
        super().__init__(local_dir)

        self.name = 'implicit_hate'

        self.group_names = [g.lower() for g in [
            'white', 'black', 'asian', 'non-white', 'latin-american', 'hispanic', 'mixed race', 'middle eastern', 'indigenous', 'race',
            'male', 'female', 'non-binary', 'trans', 'lgbtq+',
            'bisexual', 'homosexual',
            'christian', 'jewish', 'muslim/islam', 'religion',
            'physical illness/ disorder', 'mental illness/ disorder', 'physical disability', 'mental disability', 'autism', 'disability',
            'overweight', 'children', 'minors', 'old people', 'bad looking',
            'poor', 'political group', 'feminist', 'liberal', 'conservatives', 'activists', 'police',
            'violence victims', 'sexual assault/harassment victims', 'holocaust victims', 'genocide victims', 'terrorism victims', 'shooting victims', 'accident/ natural disaster victims',
            'african', 'european', 'american', 'arab', 'mexican', 'chinese', 'ethiopian', 'german', 'indian', 'japanese', 'pakistani', 'russian', 'saudis', 'southern', 'syrian',
            'blondes', 'catholic', 'immigrant', 'incest victims', 'kidnapping victims', 'minorities', 'murder victims', 'pregnant', 'priest', 'red hair', 'short people', 'slavery victims', 'war/ combat victims', 'young'
        ]]

        # Use only the 'class' column as the label
        self.class_names = ['class']

        print("Loading Implicit Hate dataset from:", local_dir)
        self.load(local_dir)
        self.prepare()

    def _set_split(self, df, split):
        self.data[split] = df['post'].to_list()

        # Since 'class' is a string label (not numeric), just store the raw values
        self.labels[split] = df[self.class_names[0]].to_numpy()

        # Encode protected groups as binary indicators
        self.protected_groups[split] = np.zeros((len(df), len(self.group_names)))
        for i, group in enumerate(self.group_names):
            self.protected_groups[split][:, i] = df['mergedMinority'].apply(
                lambda x: 1 if group in str(x).lower() else 0
            ).astype('float64')

    def load(self, local_dir=None):
        # Read the TSV file
        df = pd.read_csv(local_dir, sep='\t')

        # Skipping numeric conversion since labels are text
        # for col in self.class_names:
        #     df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing data in required columns
        df = df.dropna(subset=self.class_names + ['post', 'mergedMinority'])

        # Shuffle and split into train/dev/test (80/10/10)
        train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42),
                                             [int(0.8 * len(df)), int(0.9 * len(df))])

        self._set_split(train_df, 'train')
        self._set_split(dev_df, 'dev')
        self._set_split(test_df, 'test')
