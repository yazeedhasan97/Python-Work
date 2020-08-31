import torch.nn as nn


import constants


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def create_model():
    return ManualLinearRegression().to(constants.DEVICE)


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        Yhat = model(x)
        loss = loss_fn(y, Yhat)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step
