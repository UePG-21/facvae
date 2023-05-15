import torch
import torch.nn as nn


class FactorDecoder(nn.Module):
    """Factor decoder

    It calculates predicted stock returns `y_hat`, a random vector following the
    Gaussian distribution, which can be described by the mean `mu_y` and the covariance
    matrix `Sigma_y`, from distribution parameters of factor returns `z` (could be
    `z_post` or `z_prior`) and hidden features `e`
    """

    def __init__(self, H: int, K: int, h_alpha_size: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        K : int
            Size of factors
        h_alpha_size : int
            Size of hidden layer in alpha layer
        """
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(H, h_alpha_size)
        self.beta_layer = BetaLayer(H, K)

    def forward(
        self, e: torch.Tensor, mu_z: torch.Tensor, sigma_z: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Get distribution parameters of predicted stock returns

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H
        mu_z : torch.Tensor
            Means of posterior factor returns, B*K
        sigma_z : torch.Tensor
            Stds of posterior factor returns, B*K

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Predicted mean vector of stock returns, B*N, denoted as `mu_y`
            torch.Tensor
                Predicted cov matrix of stock returns, B*N*N, denoted as `Sigma_y`
        """
        mu_alpha, sigma_alpha = self.alpha_layer(e)
        beta = self.beta_layer(e)
        mu_y = mu_alpha + torch.einsum("bnk, bk -> bn", beta, mu_z)
        Sigma_z = torch.diag_embed(sigma_z) ** 2
        Sigma_y = torch.bmm(torch.bmm(beta, Sigma_z), beta.permute(0, 2, 1))
        Sigma_y += torch.diag_embed(sigma_alpha) ** 2
        return mu_y, Sigma_y


class AlphaLayer(nn.Module):
    """Alpha layer

    It outputs idiosyncratic returns `alpha` from the hidden features `e`

    We assume `alpha` is a Faussian random vector described by
    N(mu_alpha, diag(sigma_alpha^2))
    """

    def __init__(self, H: int, h_alpha_size: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        h_alpha_size : int
            Size of hidden layer in alpha layer
        """
        super(AlphaLayer, self).__init__()
        self.h_layer = nn.Sequential(nn.Linear(H, h_alpha_size), nn.LeakyReLU())
        self.mu_layer = nn.Linear(h_alpha_size, 1)
        self.sigma_layer = nn.Sequential(nn.Linear(h_alpha_size, 1), nn.Softplus())

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor]:
        """Get distribution parameters of idiosyncratic returns

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                means of idiosyncratic returns, B*N, denoted as `mu_alpha`
            torch.Tensor
                stds of idiosyncratic returns, B*N, denoted as `sigma_alpha`
        """
        h_alpha = self.h_layer(e)
        mu_alpha = self.mu_layer(h_alpha).squeeze(-1)
        sigma_alpha = self.sigma_layer(h_alpha).squeeze(-1)
        return mu_alpha, sigma_alpha


class BetaLayer(nn.Module):
    """Beta layer

    It calculates factor exposure `beta` from hidden feautres `e`
    """

    def __init__(self, H: int, K: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        K : int
            Size of factors
        """
        super(BetaLayer, self).__init__()
        self.bt_layer = nn.Linear(H, K)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Get factor exposures

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H

        Returns
        -------
        torch.Tensor
            Factor exposures, B*N*K, denoted as `beta`
        """
        beta = self.bt_layer(e)
        return beta


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from factor_encoder import FactorEncoder
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
    h_alpha_size = 6

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

    z = sigma_post * torch.randn(sigma_post.shape) + mu_post
    print(z.shape)

    dec = FactorDecoder(H, K, h_alpha_size)
    mu_y, Sigma_y = dec(e, mu_post, sigma_post)
    print(mu_y.shape)
    print(Sigma_y)
