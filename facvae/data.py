import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


class RollingDataset(TensorDataset):
    """Rolling dataset

    In each iteration, it yields characteristics `x` in R^{N*T*C}, and future stock
    returns `y` in R^{N}
    """

    def __init__(self, df: pd.DataFrame, label_col: str, window: int) -> None:
        """Initialization

        Parameters
        ----------
        df : pd.DataFrame
            Panel data, T_ttl*N*(C+1)
        label_col : str
            Name of the label column
        window : int
            Size of rolling window, denoted as `T`
        """
        df = df.copy()
        labels = df.pop(label_col)
        df[label_col] = labels
        self.T_ttl = df.index.get_level_values(0).nunique()
        self.N = df.index.get_level_values(1).nunique()
        L, self.C = df.shape
        if L != self.T_ttl * self.N:
            raise Exception("panel data is not balanced")
        self.start, self.end, self.T = 0, window, window
        ts = torch.from_numpy(df.values).float()
        ts = ts.reshape(self.T_ttl, self.N, self.C).permute(1, 0, 2)  # N*T_ttl*(C+1)
        self.x = ts[:, :, :-1]  # N*T_ttl*C
        self.y = ts[:, :, -1].squeeze(-1)  # N*T_ttl
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """Get item from dataset

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Characteristics, N*T*C, denoted as `x`
            torch.Tensor
                Stock returns, N, denoted as `y`
        """
        start, end = self.start + index, self.end + index
        if end > self.T_ttl:
            raise IndexError("dataset index out of range")
        return self.x[:, start:end, :], self.y[:, end - 1]

    def __len__(self) -> int:
        """Get length of the dataset

        Returns
        -------
        int
            Length of the dataset
        """
        return self.T_ttl - self.T + 1


def train_valid_test_split(
    dataset: TensorDataset, partition: list[float], shuffle: bool = False
) -> tuple[TensorDataset]:
    """Split the full dataset into training, validation, and testing dataset

    Parameters
    ----------
    dataset : TensorDataset
        Full dataset
    partition : list[float]
        Percentages of training, validation, and testing dataset
    shuffle : bool, optional
        Shuffle the full dataset before spliting or not, by default False

    Returns
    -------
    tuple[TensorDataset]
        TensorDataset
            Training dataset, denoted as `ds_train`
        TensorDataset
            Validation dataset, denoted as `ds_valid`
        TensorDataset
            Testing dataset, denoted as `ds_test`
    """
    if sum(partition) - 1.0 > 1e9 or len(partition) != 3:
        raise Exception("`partition` invalid")
    L, (pct_train, pct_valid, _) = len(dataset), partition
    indices = list(range(L))
    if shuffle:
        random.shuffle(indices)
    len_train, len_valid = int(L * pct_train), int(L * pct_valid)
    ds_train = Subset(dataset, indices[:len_train])
    ds_valid = Subset(dataset, indices[len_train : len_train + len_valid])
    ds_test = Subset(dataset, indices[len_train + len_valid :])
    return ds_train, ds_valid, ds_test


def change_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Change the frequency of the panel data

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with daily frequency
    freq : str
        Data frequency, chosen from ["w", "m", "q"] as weekly, monthly, and quarterly

    Returns
    -------
    pd.DataFrame
        Panel data with new frequency
    """
    freq = freq.lower()
    if freq not in ["w", "m", "q"]:
        raise Exception("`freq` should be chosen from ['w', 'm', 'q']")
    feature, ret = df.drop("ret", axis=1), df[["ret"]]
    df_new: pd.DataFrame = feature.groupby(level=1).resample(freq, level=0).last()
    df_new["ret"] = ret.groupby(level=1).resample(freq, level=0).sum()
    return df_new.swaplevel(0, 1, axis=0).sort_index(level=(0, 1))


def shift_ret(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift returns to the previous period then drop NaN

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with "ret" column
    periods : int, optional
        Number of the periods we want to shift return back, by default 1

    Returns
    -------
    pd.DataFrame
        Panel data with return shifted
    """
    if "ret" not in df.columns:
        raise Exception("`df` should contain a 'ret' column")

    def cs_shift(df_cs: pd.DataFrame) -> pd.DataFrame:
        """Shift returns for cross-sectional data"""
        df_cs["ret"] = df["ret"].shift(-periods)
        df_cs.iloc[-periods:] = np.nan  # force the last periods return to be NaN
        return df_cs

    return df.groupby(level=1).apply(cs_shift).dropna(axis=0)


def get_dataloaders(
    df: pd.DataFrame,
    window: int,
    batch_size: int,
    partition: list[float],
    shuffle_ds: bool = False,
    shuffle_dl: bool = False,
    drop_last: bool = True,
) -> tuple[DataLoader]:
    """Get training, validation, and testing dataloader

    Parameters
    ----------
    df : pd.DataFrame
        Panel data, T_ttl*N*(C+1), with "ret" column as future returns (shifted)
    window : int
        Size of rolling window, denoted as `T`
    batch_size : int
        How many samples per batch to load
    partition : list[float]
        Percentages of training, validation, and testing dataset
    shuffle_ds : bool, optional
        Shuffle the full dataset before spliting or not, by default False
    shuffle_dl : bool, optional
        Set to True to have the data reshuffled at every epoch, by default False
    drop_last : bool, optionl
        Set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not divisible
        by the batch size, then the last batch will be smaller, by default True

    Returns
    -------
    tuple[DataLoader]
        DataLoader
            Training dataloader, denoted as `ds_train`
        DataLoader
            Validation dataloader, denoted as `ds_valid`
        DataLoader
            Testing dataloader, denoted as `ds_test`
    """
    # get datasets
    ds_full = RollingDataset(df, window)
    ds_train, ds_valid, ds_test = train_valid_test_split(ds_full, partition, shuffle_ds)
    # get dataloaders
    dl_train = DataLoader(ds_train, batch_size, shuffle_dl, drop_last=drop_last)
    dl_valid = DataLoader(ds_valid, len(ds_valid), shuffle_dl)
    dl_test = DataLoader(ds_test, len(ds_test), shuffle_dl)
    return dl_train, dl_valid, dl_test
