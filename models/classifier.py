import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


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


class Classifier(torch.nn.Module):
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


class ClfWrapper:
    def __init__(self, model: torch.nn.Module, batch_size: int, class_weights=None, criterion=torch.nn.CrossEntropyLoss,
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
        # TODO: allow optimizer specific parameters
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

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

                batch_pred = self.model.forward(batch_x)

                loss = self.criterion(batch_pred, batch_y)
                loss.backward()

                self.optimizer.step()
                if verbose:
                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix(loss=loss.item())

                loss = loss.detach().item()

                if torch.cuda.is_available():
                    batch_pred.to('cpu')
                    batch_x = batch_x.to('cpu')
                    batch_y = batch_y.to('cpu')

                del batch_pred
                del batch_x
                del batch_y
        torch.cuda.empty_cache()

    def predict(self, X, verbose=False):
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

            batch_pred = batch_pred.detach().numpy()
            predictions.append(batch_pred)

            del batch_x

        torch.cuda.empty_cache()
        return np.vstack(predictions)


