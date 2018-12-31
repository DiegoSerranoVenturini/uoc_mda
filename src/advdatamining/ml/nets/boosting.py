import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class BoostingNet(nn.Module):

    _loss_function = None
    _optimizer = None
    optimizer = None

    def __init__(self, inputCols=None, outputCol=None, n_input=None, boosting_stack=0, hidden_layers=None, n_output=None, optimizer=None, loss_function=None):

        super(BoostingNet, self).__init__()

        self.features_names = inputCols
        self.target_name = outputCol
        self._optimizer = optimizer
        self._loss_function = loss_function

        self.boosting_fc_stack = nn.ModuleList(
            [nn.Linear(n_input, n_input) for i in range(boosting_stack)]
        )
        self.boosting_bn_stack = nn.ModuleList(
            [nn.BatchNorm1d(n_input) for i in range(boosting_stack)]
        )

        self.fc_stack = nn.ModuleList(
            [nn.Linear(n_input, hidden_layers[0])] +
            [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(0, len(hidden_layers)-1)]
        )
        self.bn_stack = nn.ModuleList(
            [nn.BatchNorm1d(hidden) for hidden in hidden_layers]
        )
        self.fc_out = nn.Linear(hidden_layers[-1], n_output)

    def forward(self, features_t):

        out = features_t
        for linear, bn in zip(self.boosting_fc_stack, self.boosting_bn_stack):
            out = linear(out)
            out = bn(out)
            out += features_t
            out = F.relu(out)

        for linear, bn in zip(self.fc_stack, self.bn_stack):
            out = linear(out)
            out = bn(out)
            out = F.relu(out)

        out = self.fc_out(out)
        out = F.log_softmax(out)

        return out

    def set_loss_function(self, loss_function):
        self._loss_function = loss_function
        return self

    def set_optimizer(self, optimizer, **args):
        if self._optimizer is None:
            self._optimizer = optimizer
        else:
            _optimizer = optimizer(**args)
            self.optimizer = _optimizer
        return self

    def build_net(self, loss_function, optimizer):
        return self.set_loss_function(loss_function).set_optimizer(optimizer)

    def fit(self, train_df, batch_size=64, num_epochs=1, init_lr=0.01):

        if not self.features_names or not self.target_name:
            raise Exception("Target and feature columns must be set before training")

        tensor_dataset = self._get_tensor_dataset(train_df)

        loader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(num_epochs):

            self.set_optimizer(self._optimizer, params=self.parameters(), lr=init_lr)

            for i, (images, labels) in enumerate(loader):
                images = Variable(images.float())
                labels = Variable(labels.long())

                # forward pass
                loss = self._forward_pass(images, labels)

                # backward pass
                loss.backward()

                # calculate the gradients
                self.optimizer.step()

                # log the losses
                losses.append(loss.item())
                if (i+1) % 100 == 0:
                    print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch+1, num_epochs, i+1, len(tensor_dataset)//batch_size, loss.data[0]))

        print(losses)
        return self

    def _get_tensor_dataset(self, train_df):

        features_df = train_df[self.features_names]
        features_tensor = torch.Tensor(features_df).float()

        target_df = train_df[self.target_name]
        target_tensor = torch.Tensor(target_df)

        return torch.utils.data.TensorDataset(features_tensor, target_tensor)

    def _forward_pass(self, features, label):
        # restart the gradient calculations
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self(features)

        # calculate the loss function
        criterion = self._loss_function()
        loss = criterion(outputs, label)

        return loss

    def predict(self, features):
        features_tensor = torch.Tensor(features)

        predictions = np.argmax(F.log_softmax(self(features_tensor)).data.numpy(), axis=1)

        return predictions

