import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


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
        # set label to be the last column
        df = df.copy()
        labels = df.pop(label_col)
        df = pd.concat([df, labels], axis=1)
        # get attributes
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
        # to device
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


def change_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Change the frequency of the panel data

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with daily frequency
    freq : str
        Data frequency, chosen from ["d", "w", "m", "q"]

    Returns
    -------
    pd.DataFrame
        Panel data with new frequency
    """
    freq = freq.lower()
    if freq not in ["d", "w", "m", "q"]:
        raise Exception("`freq` should be chosen from ['d', 'w', 'm', 'q']")
    if freq == "d":
        return df.copy()
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
    df["ret"] = df["ret"].groupby(level=1).shift(-periods)
    return df.dropna()


def wins_ret(df: pd.DataFrame, wins_thresh: float) -> pd.DataFrame:
    """Winsorize returns

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with "ret" column
    wins_thresh : float
        Threshold of winsorization

    Returns
    -------
    pd.DataFrame
        Panel data with return winsorized
    """

    def wins(x: np.ndarray, l: float, u: float) -> np.ndarray:
        """Winsorize x in [l, u]

        Parameters
        ----------
        x : np.ndarray
            Data to be winsorized
        l : float
            Lower bound
        u : float
            Upper bound

        Returns
        -------
        np.ndarray
            Winsorized values
        """
        return np.where(x < l, l, np.where(x > u, u, x))

    if "ret" not in df.columns:
        raise Exception("`df` should contain a 'ret' column")
    df = df.copy()
    df["ret"] = wins(df["ret"], -wins_thresh, wins_thresh)
    return df


def assign_label(
    df: pd.DataFrame, quantiles: list[float]
) -> tuple[pd.DataFrame, pd.Series]:
    """Assign labels based on different quantiles of the returns

    Parameters
    ----------
    df : float
        Panel data with "ret" column
    quantiles : float
        Quantiles to block returns

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        pd.DataFrame
            Panel data with "label" in the last column but "ret" removed
        pd.Series
            Removed "ret" column
    """

    def ret_to_label(ret: float, threshs: list[float]) -> float:
        """Convert return to label

        Parameters
        ----------
        ret : float
            Return value
        h_thresh : float
            Higher threshold
        l_thresh : float
            Lower threshold

        Returns
        -------
        float
            Label value
        """
        max_label = len(threshs) // 2
        for i, th in enumerate(threshs):
            if ret < th:
                return i - max_label
        return max_label

    def cs_assign(cs_ret: pd.Series) -> pd.Series:
        """Cross-sectionally assign label

        Parameters
        ----------
        cs_ret : pd.Series
            Cross-sectional return (N*1)

        Returns
        -------
        pd.Series
            Cross-sectional label (N*1)
        """
        threshs = cs_ret.quantile(quantiles)
        return cs_ret.apply(lambda x: ret_to_label(x, threshs))

    quantiles.sort()
    if quantiles[0] < 0 or quantiles[-1] > 0.5:
        raise Exception("element in `quantiles` out of range")
    quantiles.extend([1 - q for q in reversed(quantiles)])
    df = df.copy()
    df["label"] = df["ret"].groupby(level=0).apply(cs_assign)
    return df.drop("ret", axis=1), df["ret"]
