from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def create_dataset(x_tensor, y_tensor):
    return CustomDataset(x_tensor, y_tensor)


def create_train_val_dataset(dataset, sizes=None):
    if sizes is None:
        sizes = [80, 20]
    return random_split(dataset, sizes)
