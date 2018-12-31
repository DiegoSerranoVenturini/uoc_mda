import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CNN(nn.Module):

    _loss_function = None
    _optimizer = None
    optimizer = None

    def __init__(self, n_hidden, n_output, inputCols=None, outputCol=None, optimizer=None, loss_function=None):
        super(CNN, self).__init__()

        self.features_names = inputCols
        self.target_name = outputCol
        self._optimizer = optimizer
        self._loss_function = loss_function

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=5, padding=2),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(n_hidden, n_hidden*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(n_hidden*2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.linear_out = nn.Linear(7 * 7 * n_hidden*2, n_output)

    def forward(self, features_tensor):
        out = self.conv_1(features_tensor)

        out = self.conv_2(out)

        out = out.view(out.size(0), -1)

        out = self.linear_out(out)

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

    def setInputCol(self, inputCols):
        self.features_names = inputCols

    def setOutputCol(self, outputCol):
        self.target_name = outputCol

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
        features_tensor = torch.Tensor(features_df.reshape(-1, 1, 28, 28)).float()

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
        features_tensor = torch.Tensor(features.reshape(-1, 1, 28, 28))

        predictions = np.argmax(F.log_softmax(self(features_tensor)).data.numpy(), axis=1)

        return predictions


class WeightedCNN(CNN):

    WEIGHTS = torch.Tensor([1, 2, 1, 1, 1])

    def _forward_pass(self, features, label):
        # restart the gradient calculations
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self(features)

        # calculate the loss function
        loss = nn.CrossEntropyLoss(weight=self.WEIGHTS)

        loss = loss(outputs, label)

        return loss


