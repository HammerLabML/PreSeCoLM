import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from salsa.SaLSA import SaLSA
import prototorch as pt

class BertLikeClassifier(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):

        super().__init__()
        # this imitates the BERT clf head (including the pooler) + sigmoid layer
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, input_size)
        self.activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available():
            self.linear1 = self.linear1.to('cuda')
            self.activation = self.activation.to('cuda')
            self.linear2 = self.linear2.to('cuda')
            self.sigmoid = self.sigmoid.to('cuda')

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class MLP2Layer(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available():
            self.linear1 = self.linear1.to('cuda')
            self.activation = self.activation.to('cuda')
            self.linear2 = self.linear2.to('cuda')
            self.sigmoid = self.sigmoid.to('cuda')

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def to_cpu(self):
        self.linear1 = self.linear1.to('cpu')
        self.activation = self.activation.to('cpu')
        self.linear2 = self.linear2.to('cpu')
        self.sigmoid = self.sigmoid.to('cpu')


class MLP3Layer(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size1: int, hidden_size2: int):
        super().__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size1)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size2, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available():
            self.linear1 = self.linear1.to('cuda')
            self.activation1 = self.activation1.to('cuda')
            self.linear2 = self.linear2.to('cuda')
            self.activation2 = self.activation2.to('cuda')
            self.linear3 = self.linear3.to('cuda')
            self.sigmoid = self.sigmoid.to('cuda')

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

    def to_cpu(self):
        self.linear1 = self.linear1.to('cpu')
        self.activation1 = self.activation1.to('cpu')
        self.linear2 = self.linear2.to('cpu')
        self.activation2 = self.activation2.to('cpu')
        self.linear3 = self.linear3.to('cpu')
        self.sigmoid = self.sigmoid.to('cpu')


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available():
            self.linear1 = self.linear1.to('cuda')
            self.sigmoid = self.sigmoid.to('cuda')

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x

    def to_cpu(self):
        self.linear1 = self.linear1.to('cpu')
        self.sigmoid = self.sigmoid.to('cpu')


class PrototorchWrapper:
    def __init__(self, model: torch.nn.Module, batch_size: int, class_weights=None, criterion=torch.nn.BCEWithLogitsLoss,
                 optimizer=torch.optim.RMSprop, lr=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.criterion = pt.losses.GLVQLoss()

        # the optimizer will be  initialized in fit(),
        # because the prototorch head needs data to finish initializiation
        self.optimizer = None
        self.optimizer_class = optimizer
        self.lr = lr

    def fit_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                           max_epochs: int, delta: float = 0.0, patience: int = 10, verbose: bool = False):
        print("early stopping not supported for ProtoTorchPipeline, use regular fit() function")
        self.fit(X_train, y_train, epochs=max_epochs, verbose=verbose)
        return max_epochs

    def fit(self, X, y, epochs, verbose=False):
        # init LVQ prototypes with the data
        self.model.init_data(TensorDataset(torch.tensor(X), torch.tensor(y)))

        # LVQ implementation has their own fit function
        self.model.fit(X, y, self.optimizer_class, self.lr, self.criterion, batch_size=self.batch_size,
                       epochs=epochs, verbose=verbose, init_head=True)

    def predict(self, X, verbose=False):
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        pred_outputs = []

        if verbose:
            loop = tqdm(loader, leave=True)
        else:
            loop = loader

        for batch in loop:
            batch_x = batch[0]
            if torch.cuda.is_available():
                batch_x = batch_x.to('cuda')

            pred = self.model.predict(batch_x)
            pred_outputs.append(pred.to('cpu').detach().numpy())

            batch_x = batch_x.to('cpu')
            del batch_x
            torch.cuda.empty_cache()

        # handle 1D output from LQV
        if pred_outputs[0].ndim == 1:
            pred_stack = np.vstack([np.expand_dims(pred, axis=1) for pred in pred_outputs]).squeeze()
        else:
            pred_stack = np.vstack(pred_outputs)

        return pred_stack


class ClfWrapper:
    def __init__(self, model: torch.nn.Module, batch_size: int, class_weights=None, criterion=torch.nn.BCEWithLogitsLoss,
                 optimizer=torch.optim.RMSprop, lr=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.class_weights = class_weights
        if class_weights is not None:
            print("use class weights")
            class_weights = torch.tensor(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.to('cuda')
            self.criterion = criterion(pos_weight=class_weights)
        else:
            self.criterion = criterion()

        if optimizer is SaLSA:
            self.optimizer = optimizer(self.model.parameters())
        else:
            assert lr is not None
            self.optimizer = optimizer(self.model.parameters(), lr=lr)

    def fit_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                           max_epochs: int, delta: float = 0.0, patience: int = 10, verbose: bool = False):
        assert max_epochs > 2
        best_loss = 999
        counter = 0
        final_epochs = max_epochs
        patience = min(patience, max_epochs - 2)  # otherwise early stopping doesn't make sense
        if verbose:
            print("run early stopping with delta=%f and patience=%i" % (delta, patience))

        y_val = torch.from_numpy(y_val)

        for i in range(max_epochs):
            self.fit(X=X_train, y=y_train, epochs=1, verbose=verbose)
            pred = self.predict(X_val, return_tensors=True)

            if torch.cuda.is_available():
                y_val = y_val.to('cuda')
                pred = pred.to('cuda')

            val_loss = self.criterion(pred, y_val)

            if torch.cuda.is_available():
                pred = pred.to('cpu')
                y_val = y_val.to('cpu')
                val_loss = val_loss.to('cpu')

            if val_loss < best_loss - delta:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter == patience:
                final_epochs = i
                break

        if final_epochs < max_epochs:
            print("stopping after %i epochs with validation loss %.3f" % (i, val_loss))
        return final_epochs

    def fit(self, X, y, epochs, verbose=False):
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
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

                def closure(backwards=False):
                    batch_pred = self.model.forward(batch_x)
                    loss = self.criterion(batch_pred, batch_y)
                    if backwards:
                        loss.backward()

                    batch_pred.to('cpu')
                    del batch_pred

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
        torch.cuda.empty_cache()

    def predict(self, X, verbose=False, return_tensors=False):
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []

        if verbose:
            loop = tqdm(loader, leave=True)
        else:
            loop = loader

        for batch in loop:
            if torch.cuda.is_available():
                batch_x = batch[0].to('cuda')

            batch_pred = self.model.forward(batch_x)

            if torch.cuda.is_available():
                batch_pred = batch_pred.to('cpu')
                batch_x = batch_x.to('cpu')

            if return_tensors:
                predictions.append(batch_pred.detach())
            else:
                batch_pred = batch_pred.detach().numpy()
                predictions.append(batch_pred)

            del batch_x

        torch.cuda.empty_cache()

        if return_tensors:
            predictions = torch.vstack(predictions)
        else:
            predictions = np.vstack(predictions)

        return predictions



