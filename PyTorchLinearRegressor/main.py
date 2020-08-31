import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchviz import make_dot
import constants
import datasets

# np.random.seed(42)
# x = np.random.rand(100, 1)
from model import create_model, make_train_step

true_a, true_b = 1, 2
# y = true_a + true_b * x + 0.1 * np.random.randn(100, 1)
# x_tensor = torch.from_numpy(x).float()
# y_tensor = torch.from_numpy(y).float()

torch.manual_seed(constants.RANDOM_SEED)
x_tensor = torch.randn(100, 1, requires_grad=True, dtype=torch.float)
y_tensor = true_a + true_b * x_tensor + 0.1 * torch.randn(100, 1, requires_grad=True, dtype=torch.float)

print(x_tensor)
print(y_tensor)

print("-" * 75)
train_dataset, val_dataset = datasets.create_train_val_dataset(datasets.create_dataset(x_tensor, y_tensor), [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=constants.BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=constants.BATCH_SIZE)

model = create_model()
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-3)
train_step = make_train_step(model, loss_fn, optimizer)

training_losses = []
validation_losses = []
print(model.state_dict())

for epoch in range(constants.N_EPOCHS):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(constants.DEVICE)
        y_batch = y_batch.to(constants.DEVICE)
        loss = train_step(x_batch, y_batch)
        batch_losses.append(loss)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(constants.DEVICE)
            y_val = y_val.to(constants.DEVICE)
            model.eval()
            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat).item()
            val_losses.append(val_loss)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"[{epoch + 1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

print(model.state_dict())
