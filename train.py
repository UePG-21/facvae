import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


class RollingDataset(TensorDataset):
    """Rolling dataset

    In each iteration, it yields characteristics `x` in R^{N*T*C}, and future stock
    returns `y` in R^{N}
    """

    def __init__(self, df: pd.DataFrame, window: int) -> None:
        """Initialization

        Parameters
        ----------
        df : pd.DataFrame
            Panel data, T_ttl*N*(C+1), with "ret" column as future returns (shifted)
            When we mention panel data, it should satis  # TODO: asdf
        window : int
            Size of rolling window, denoted as `T`
        """
        self.start, self.end = 0, window
        # put "ret" in the last column
        df = df.copy()
        ret = df.pop("ret")
        df["ret"] = ret
        # sort by time index
        df.sort_index(level=0, inplace=True)
        # load as tensor
        ts = torch.from_numpy(df.values)  # T_ttl*N*(C+1)
        ts = ts.permute(1, 0, 2)  # N*T_ttl*(C+1)
        self.x = ts[:, :, :-1]  # N*T_ttl*C
        self.y = ts[:, :, -1]  # N*T_ttl
        
    def __getitem__(self, index):
        pass
        
        

def train_valid_test_split():
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    B = 32