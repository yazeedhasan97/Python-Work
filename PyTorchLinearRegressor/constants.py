import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 500
BATCH_SIZE = 16

RANDOM_SEED = 42
