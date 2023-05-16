import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader




def train_valid_test_split():
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    B = 32