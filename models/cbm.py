from tqdm import tqdm

import numpy as np

from .classifier import Classifier, LinearClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader


class CBM(torch.nn.Module):
    # the model transforms the input activations through two concept layers:
    # C1 the specified concepts, for which a loss is computed based on concept labels
    # C2 the unspecified concepts, that the model optimized only based on the clf loss
    # both outputs are concatenated and passed through the clf head

    def __init__(self, input_size, output_size, n_learned_concepts, n_other_concepts, hidden_size=None):
        super().__init__()
        self.XtoC1 = torch.nn.Linear(input_size, n_learned_concepts)
        self.XtoC2 = torch.nn.Linear(input_size, n_other_concepts)

        if torch.cuda.is_available():
            self.XtoC1 = self.XtoC1.to('cuda')
            self.XtoC2 = self.XtoC2.to('cuda')

        n_concepts = n_learned_concepts + n_other_concepts
        if hidden_size is None:
            self.clf = LinearClassifier(n_concepts, output_size)
        else:
            self.clf = Classifier(n_concepts, output_size, hidden_size)

    def forward(self, x):
        c1 = self.XtoC1(x)
        c2 = self.XtoC2(x)
        c = torch.cat((c1, c2), dim=1)
        out = self.clf.forward(c)
        output = [out, c1]
        return output

class CBMWrapper():
    
    def __init__(self, model: torch.nn.Module, batch_size: int, class_weights=None, criterion=torch.nn.CrossEntropyLoss,
                 optimizer=torch.optim.RMSprop, lr=1e-3, lambda_concept=0.5, concept_criterion=torch.nn.CrossEntropyLoss,
                 concept_weights=None):
        self.model = model
        self.batch_size = batch_size
        self.class_weights = class_weights
        if class_weights is not None:
            print("use class weights")
            print(class_weights)
            class_weights = torch.tensor(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.to('cuda')
            self.criterion = criterion(pos_weight=class_weights)
        else:
            self.criterion = criterion()

        if concept_weights is not None:
            print("use class weights")
            concept_weights = torch.tensor(concept_weights)
            if torch.cuda.is_available():
                concept_weights = concept_weights.to('cuda')
            self.concept_criterion = concept_criterion(pos_weight=concept_weights)
        else:
            self.concept_criterion = concept_criterion()
            
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.lambda_concept = lambda_concept

    def fit(self, X, y, concepts, epochs, verbose=False):
        self.model.train()
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y), torch.tensor(concepts))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            if verbose:
                loop = tqdm(loader, leave=True)
            else:
                loop = loader

            for batch in loop:
                self.optimizer.zero_grad()

                if torch.cuda.is_available():
                    batch_x = batch[0].to('cuda')
                    batch_y = batch[1].to('cuda')
                    batch_c = batch[2].to('cuda')

                out = self.model.forward(batch_x)
                pred = out[0]
                concepts = out[1]

                pred_loss = self.criterion(pred, batch_y)
                concept_loss = self.concept_criterion(concepts, batch_c)
                loss = self.lambda_concept*concept_loss + pred_loss
                loss.backward()
                self.optimizer.step()
                if verbose:
                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix(loss=loss.item())

                loss = loss.detach().item()

                if torch.cuda.is_available():
                    pred = pred.to('cpu')
                    concepts = concepts.to('cpu')
                    batch_x = batch_x.to('cpu')
                    batch_y = batch_y.to('cpu')

                del out
                del batch_x
                del batch_y
        self.model.eval()
        torch.cuda.empty_cache()

    def predict(self, X, verbose=False, batch_size=64):
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = []
        all_concepts = []

        if verbose:
            loop = tqdm(loader, leave=True)
        else:
            loop = loader

        for batch in loop:
            if torch.cuda.is_available():
                batch_x = batch[0].to('cuda')

            out = self.model.forward(batch_x)
            pred = out[0]
            concepts = out[1]

            if torch.cuda.is_available():
                pred = pred.to('cpu')
                concepts = concepts.to('cpu')
                batch_x = batch_x.to('cpu')

            pred = pred.detach().numpy()
            predictions.append(pred)
            concepts = concepts.detach().numpy()
            all_concepts.append(concepts)

            del batch_x

        torch.cuda.empty_cache()
        return np.vstack(predictions), np.vstack(all_concepts)