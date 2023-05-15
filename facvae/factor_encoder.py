import torch
import torch.nn as nn


class FactorEncoder(nn.Module):
    """Factor encoder

    It extracts posterior factors `z_post` is a random vector following the independent
    Gaussian distribution, which can be descibed by the mean `mu_post` and the standard
    deviation `sigma_post`
    """

    def __init__(self, H: int, M: int, K: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        M : int
            Size of portfolios
        K : int
            Size of factors
        """
        super(FactorEncoder, self).__init__()
        self.portfolio_layer = PortfolioLayer(H, M)
        self.mapping_layer = MappingLayer(M, K)

    def forward(self, e: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Get distribution parameters of posterior factor returns

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*H
        y : torch.Tensor
            Stock returns, B*N

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Means of posterior factor returns, B*K, denoted as `mu_post`
            torch.Tensor
                Stds of posterior factor returns, B*K, denoted as `sigma_post`
        """
        a = self.portfolio_layer(e)
        y_p = torch.einsum("bn, bnm -> bm", y, a)
        mu_post, sigma_post = self.mapping_layer(y_p)
        return mu_post, sigma_post


class PortfolioLayer(nn.Module):
    """Portfolio layer

    It dynamically re-weights the portfolios on the basis of stock hidden features

    The main advantages of constructing portfolios lie in: 1) reducing the input
    dimension and avoiding too many parameters; 2) robust to the missing stocks in
    cross-section and thus suitable for the market
    """

    def __init__(self, H: int, M: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        M : int
            Size of portfolios
        """
        super(PortfolioLayer, self).__init__()
        self.pf_layer = nn.Sequential(nn.Linear(H, M), nn.Softmax(dim=1))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Get portfolio weights

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H

        Returns
        -------
        torch.Tensor
            Portfolio weights, B*N*M, denoted as `a`
        """
        a = self.pf_layer(e)
        return a


class MappingLayer(nn.Module):
    """Mapping layer

    It maps `y_p` as the portfolio returns to the distribution of posterior factor
    returns `z_post`
    """

    def __init__(self, M: int, K: int) -> None:
        """Initialization

        Parameters
        ----------
        M : int
            Size of portfolios
        K : int
            Size of factors
        """
        super(MappingLayer, self).__init__()
        self.mean_layer = nn.Linear(M, K)
        self.std_layer = nn.Sequential(nn.Linear(M, K), nn.Softplus())

    def forward(self, y_p: torch.Tensor) -> tuple[torch.Tensor]:
        """Get mean and std of the posterior factor returns

        Parameters
        ----------
        y_p : torch.Tensor
            Portfolio returns, B*M

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Means of posterior factor returns, B*K, denoted as `mu_post`
            torch.Tensor
                Stds of posterior factor returns, B*K, denoted as `sigma_post`
        """
        mu_post = self.mean_layer(y_p)
        sigma_post = self.std_layer(y_p)
        return mu_post, sigma_post


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from feature_extractor import FeatureExtractor

    dir_main = "E:/Others/Programming/py_vscode/projects/signal_mixing/"
    dir_code = dir_main + "code/"
    dir_data = dir_main + "data/"
    dir_config = dir_main + "config/"
    dir_result = dir_main + "result/"

    H = 5
    N = 74
    T = 20
    C = 10
    M = 18
    K = 9
    h_proj_size = 6

    df_l1 = pd.read_pickle(dir_data + "df_l1_comb.pickle").sort_index(level=0)
    x = (
        df_l1.loc[:"2015-01-30", :"maxretpayoff"]
        .values.reshape(T, N, C)
        .transpose(1, 0, 2)
        .reshape(1, N, T, C)
    )
    x = torch.tensor(x).float()
    y = df_l1.loc["2015-01-30", "ret"].values.reshape(1, N)
    y = torch.tensor(y).float()
    fe = FeatureExtractor(N, T, C, H, h_proj_size)
    e = fe(x)

    enc = FactorEncoder(H, M, K)
    mu_post, sigma_post = enc(e, y)
    print(mu_post.shape)
    print(sigma_post.shape)
