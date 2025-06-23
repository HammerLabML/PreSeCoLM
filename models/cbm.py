from tqdm import tqdm
import numpy as np
from .classifier import MLP2Layer, LinearClassifier, MLP3Layer
import torch
from torch.utils.data import TensorDataset, DataLoader
from salsa.SaLSA import SaLSA


class CBM(torch.nn.Module):
    # the model transforms the input activations through two concept layers:
    # C1 the specified concepts, for which a loss is computed based on concept labels
    # C2 the unspecified concepts, that the model optimized only based on the clf loss
    # both outputs are concatenated and passed through the clf head

    def __init__(self, input_size, output_size, n_concepts_protec, n_concepts_unsup,
                 hidden_size=None, hidden_size2=None):
        super().__init__()
        self.XtoC1 = torch.nn.Linear(input_size, n_concepts_protec)
        self.XtoC2 = torch.nn.Linear(input_size, n_concepts_unsup)

        if torch.cuda.is_available():
            self.XtoC1 = self.XtoC1.to('cuda')
            self.XtoC2 = self.XtoC2.to('cuda')

        n_concepts = n_concepts_protec + n_concepts_unsup

        if hidden_size is not None and hidden_size2 is not None:
            print("use a linear classifier")
            self.clf = MLP3Layer(n_concepts, output_size, hidden_size, hidden_size2)
        if hidden_size is not None:
            print("use a linear classifier")
            self.clf = MLP2Layer(n_concepts, output_size, hidden_size)
        else:
            print("use a linear classifier")
            self.clf = LinearClassifier(n_concepts, output_size)

    def forward(self, x):
        c1 = self.XtoC1(x)
        c2 = self.XtoC2(x)
        c = torch.cat((c1, c2), dim=1)
        out = self.clf.forward(c)
        output = [out, c1]
        return output

    def to_cpu(self):
        self.XtoC1 = self.XtoC1.to('cpu')
        self.XtoC2 = self.XtoC2.to('cpu')
        self.clf.to_cpu()


class CBMNonLinear(torch.nn.Module):
    # the model transforms the input activations through two concept layers:
    # C1 the specified concepts, for which a loss is computed based on concept labels
    # C2 the unspecified concepts, that the model optimized only based on the clf loss
    # both outputs are concatenated and passed through the clf head

    def __init__(self, input_size, output_size, n_learned_concepts, n_other_concepts, hidden_size=None):
        super().__init__()
        self.XtoC1 = torch.nn.Linear(input_size, n_learned_concepts)
        self.XtoC2 = torch.nn.Linear(input_size, n_other_concepts)
        self.activation = torch.nn.ReLU()
        self.C1toC1b = torch.nn.Linear(n_other_concepts, n_other_concepts)
        self.C2toC2b = torch.nn.Linear(n_other_concepts, n_other_concepts)

        if torch.cuda.is_available():
            self.XtoC1 = self.XtoC1.to('cuda')
            self.XtoC2 = self.XtoC2.to('cuda')
            self.activation = self.XtoC2.to('cuda')
            self.C1toC1b = self.XtoC2.to('cuda')
            self.C2toC2b = self.XtoC2.to('cuda')

        n_concepts = n_learned_concepts + n_other_concepts
        if hidden_size is None:
            self.clf = LinearClassifier(n_concepts, output_size)
        else:
            self.clf = MLP2Layer(n_concepts, output_size, hidden_size)

    def forward(self, x):
        # supervised concepts
        c1 = self.XtoC1(x)
        c1 = self.activation(c1)
        c1 = self.C1toC1b(c1)

        # other concepts
        c2 = self.XtoC2(x)
        c2 = self.activation(c2)
        c2 = self.C2toC2b(c2)

        # join concepts
        c = torch.cat((c1, c2), dim=1)

        # classify
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

        if optimizer is SaLSA:
            self.optimizer = optimizer(self.model.parameters())
        else:
            assert lr is not None
            self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.lambda_concept = lambda_concept

    def fit_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray, c_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray, c_val: np.ndarray,
                                max_epochs: int, delta: float = 0.0, patience: int = 10, verbose: bool = False):
        assert max_epochs > 2
        best_loss = 999
        best_concept_loss = 999
        counter = 0
        final_epochs = max_epochs
        patience = min(patience, max_epochs - 2)  # otherwise early stopping doesn't make sense
        if verbose:
            print("run early stopping with delta=%f and patience=%i" % (delta, patience))

        y_val = torch.from_numpy(y_val)
        c_val = torch.from_numpy(c_val)

        for i in range(max_epochs):
            self.fit(X=X_train, y=y_train, concepts=c_train, epochs=1, verbose=verbose)
            pred, concepts = self.predict(X_val, return_tensors=True)

            if torch.cuda.is_available():
                y_val = y_val.to('cuda')
                c_val = c_val.to('cuda')
                pred = pred.to('cuda')
                concepts = concepts.to('cuda')
                concepts = concepts.to('cuda')

            pred_loss = self.criterion(pred, y_val)
            concept_loss = self.concept_criterion(concepts, c_val)

            if torch.cuda.is_available():
                concepts = concepts.to('cpu')
                pred = pred.to('cpu')
                y_val = y_val.to('cpu')
                c_val = c_val.to('cpu')
                pred_loss = pred_loss.to('cpu')
                concept_loss = concept_loss.to('cpu')

            del concepts
            del pred

            # continue as long as one loss improves
            if pred_loss < best_loss - delta:
                best_loss = pred_loss
                counter = 0
            elif concept_loss < best_concept_loss - delta:
                best_concept_loss = concept_loss
                counter = 0
            else:
                counter += 1

            if counter == patience:
                final_epochs = i
                break

        if final_epochs < max_epochs:
            print("stopping after %i epochs with validation loss %.3f and concept loss %.3f" % (i, pred_loss, concept_loss))
        return final_epochs

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

                def closure(backwards=False):
                    out = self.model.forward(batch_x)
                    pred = out[0]
                    concepts = out[1]

                    pred_loss = self.criterion(pred, batch_y)
                    concept_loss = self.concept_criterion(concepts, batch_c)
                    loss = self.lambda_concept * concept_loss + pred_loss
                    if backwards:
                        loss.backward()

                    pred.to('cpu')
                    concepts.to('cpu')
                    del pred
                    del concepts

                    return loss

                loss = self.optimizer.step(closure=closure)

                if verbose:
                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix(loss=loss.item())

                loss = loss.detach().item()

                if torch.cuda.is_available():
                    batch_x = batch_x.to('cpu')
                    batch_y = batch_y.to('cpu')

                del batch_x
                del batch_y
        self.model.eval()
        torch.cuda.empty_cache()

    def predict(self, X, verbose=False, batch_size=64, return_tensors=False):
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

            if not return_tensors:
                pred = pred.detach().numpy()
                concepts = concepts.detach().numpy()
            predictions.append(pred)
            all_concepts.append(concepts)

            del batch_x

        torch.cuda.empty_cache()

        if return_tensors:  # for early stopping
            return torch.vstack(predictions), torch.vstack(all_concepts)
        else:
            return np.vstack(predictions), np.vstack(all_concepts)
