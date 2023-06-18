"""
Reimplementation of "FactorVAE: A Probabilistic Dynamic Factor Model Based on
Variational Autoencoder for Predicting Cross-Sectional Stock Returns"

Author: UePG

Reference:
[1] https://ojs.aaai.org/index.php/AAAI/article/view/20369

Coding convention:
1) Panel data
    When we mention panel data, it is of type pd.DataFrame and should satisfy the
    following conditions:
    - having MultiIndex with time periods and samples in order
    - having time period index whose value is of type pd.Timestamp
    - sorted by time period index with sample order fixed in each time period
    - having "ret" column if and only if "ret" means return and it is the last column
    - balanced if not otherwise specified
    - having no missing value if not otherwise specified
2) Letter case
    Lowercase letters are used to denote vectors or matrices (tensor); capital letters
    are used to denote scalars (constant)
"""

__version__ = "0.2.2"

__all__ = ["data", "pipeline", "backtesting"]


from typing import Any, Callable

import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from .backtesting import Backtester
from .factor_decoder import FactorDecoder
from .factor_encoder import FactorEncoder
from .factor_predictor import FactorPredictor
from .feature_extractor import FeatureExtractor
from .pipeline import Pipeline


class FactorVAE(nn.Module):
    """Factor VAE (Variational Auto-Encoder)

    It extracts effective factors from noisy market data. First, it obtain optimal
    factors by an encoder-decoder architecture with access to future data, and then
    train a factor predictor according a prior-posterior learning method, which extracts
    factors to approximate the optimal factors.

    Notation
    - Scalar (constant)
        - `B`: size of batch (arbitrary)
        - `N`: size of stocks (arbitrary)
        - `T`: size of time periods (arbitrary)
        - `C`: size of characteristics
        - `H`: size of hidden features
        - `M`: size of portfolios
        - `K`: size of factors
    - Tensor (variable)
        - `x`: characteristics, B*N*T*C
        - `y`: stock returns, B*N
        - `z_post`: posterior latent factor returns, B*K
        - `z_prior`: prior latent factor returns, B*K
        - `y_hat`: reconstructed stock returns, B*N
        - `e`: hidden features, B*N*H
        - `mu_post`: mean vector of `z_post`, B*K
        - `sigma_post`: std vector of `z_post`, B*K
        - `mu_prior`: mean vector of `z_prior`, B*K
        - `sigma_prior`: std vector of `z_prior`, B*K
        - `mu_y`: mean vector of `y_hat`, B*N
        - `Sigma_y`: cov matrix of `y_hat`, B*N*N
    - Distribution
        - p_{theta}(y|x): true label, likelihood
        - q_{phi}(z|x,y): encoder output, posterior distribution
        - q_{phi}(z|x): predictor output, prior distribution
        - p_{theta}(y|x,z): decoder output, conditional likelihood
        - f_{phi,theta}(y|x): predicted label, predicted likelihood
    """

    def __init__(
        self,
        C: int,
        H: int,
        M: int,
        K: int,
        h_proj_size: int,
        h_alpha_size: int,
        h_prior_size: int,
    ) -> None:
        """Initialization

        Parameters
        ----------
        C : int
            Size of characteristics
        H : int
            Size of hidden features
        M : int
            Size of portfolios
        K : int
            Size of factors
        h_proj_size : int
            Size of hidden layer in projection layer
        h_alpha_size : int
            Size of hidden layer in alpha layer
        h_prior_size : int
            Size of hidden layer in prior layer
        """
        super(FactorVAE, self).__init__()
        self.extractor = FeatureExtractor(C, H, h_proj_size)
        self.encoder = FactorEncoder(H, M, K)
        self.decoder = FactorDecoder(H, K, h_alpha_size)
        self.predictor = FactorPredictor(H, K, h_prior_size)

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterize the sampling layer

        Parameters
        ----------
        mu : torch.Tensor
            Means, B*K
        sigma : torch.Tensor
            Stds, B*K

        Returns
        -------
        torch.Tensor
            Multivariable sampled from N(mu, diag(sigma))
        """
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Get distribution parameters of `y_hat`, `z_post`, and `z_prior`

        Parameters
        ----------
        x : torch.Tensor
            Characteristics, B*N*T*C
        y : torch.Tensor
            Stock returns, B*N

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Predicted mean vector of stock returns, B*N, denoted as `mu_y`
            torch.Tensor
                Predicted std vector of stock returns, B*N, denoted as `sigma_y`
            torch.Tensor
                Means of posterior factor returns, B*K, denoted as `mu_post`
            torch.Tensor
                Stds of posterior factor returns, B*K, denoted as `sigma_post`
            torch.Tensor
                Means of prior factor returns, B*K, denoted as `mu_prior`
            torch.Tensor
                Stds of prior factor returns, B*K, denoted as `sigma_prior`
        """
        e = self.extractor(x)
        mu_post, sigma_post = self.encoder(e, y)
        z_post = self.reparameterize(mu_post, sigma_post)
        mu_y, sigma_y = self.decoder(e, z_post)
        mu_prior, sigma_prior = self.predictor(e)
        return mu_y, sigma_y, mu_post, sigma_post, mu_prior, sigma_prior

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Predict stock returns from stock characteristics

        Parameters
        ----------
        x : torch.Tensor
            Characteristics, B*N*T*C

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Predicted mean vector of stock returns, B*N, denoted as `mu_y`
            torch.Tensor
                Predicted cov matrix of stock returns, B*N*N, denoted as `Sigma_y`
        """
        e = self.extractor(x)
        mu_prior, sigma_prior = self.predictor(e)
        mu_y, Sigma_y = self.decoder.predict(e, mu_prior, sigma_prior)
        return mu_y, Sigma_y


def loss_fn_vae(
    y: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_y: torch.Tensor,
    mu_post: torch.Tensor,
    sigma_post: torch.Tensor,
    mu_prior: torch.Tensor,
    sigma_prior: torch.Tensor,
    gamma: float = 1.0,
    lmd: float = 0.5,
) -> torch.Tensor:
    """Loss function of FactorVAE

    Parameters
    ----------
    y : torch.Tensor
        Stock returns, B*N
    mu_y : torch.Tensor
        Predicted mean vector of stock returns, B*N
    sigma_y : torch.Tensor
        Predicted std vector of stock returns, B*N
    mu_post : torch.Tensor
        Means of posterior factor returns, B*K
    sigma_post : torch.Tensor
        Stds of posterior factor returns, B*K
    mu_prior : torch.Tensor
        Means of prior factor returns, B*K
    sigma_prior : torch.Tensor
        Stds of prior factor returns, B*K
    gamma : float, optional
        Gamma as regularization parameter, by default 2.0
    lmd : float, optional
        Lambda as mixing parameter, by default 0.5

    Returns
    -------
    torch.Tensor
        Loss values, B, denoted as `loss`
    """
    ic = bcorr(y, mu_y)
    dist_y = Normal(mu_y, sigma_y)
    ll = dist_y.log_prob(y).sum(-1)
    kld = gaussian_kld(mu_post, mu_prior, sigma_post, sigma_prior)
    loss = -ic + gamma * ((1-lmd) * -ll + lmd * kld) / y.shape[1]
    return loss


def bcorr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch correlation between two vectors

    Parameters
    ----------
    x : torch.Tensor
        The first tensor, B*N
    y : torch.Tensor
        The sencond tensor, B*N

    Returns
    -------
    torch.Tensor
        Correlation coefficient, B, denoted as `corr`
    """
    x = x - x.mean(dim=1).unsqueeze(-1)
    y = y - y.mean(dim=1).unsqueeze(-1)
    cov = torch.einsum("bn, bn -> b", x, y) / x.shape[1]
    x_std, y_std = x.std(dim=1), y.std(dim=1)
    corr = cov / x_std / y_std
    return corr


def gaussian_kld(
    mu1: torch.Tensor, mu2: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """KL divergence of two multivariate independent Gaussian distributions

    Parameters
    ----------
    mu1 : torch.Tensor
        Means of the first Gaussian, B*K
    mu2 : torch.Tensor
        Means of the second Gaussian, B*K
    sigma1 : torch.Tensor
        Stds of the first Gaussian, B*K
    sigma2 : torch.Tensor
        Stds of the second Gaussian, B*K

    Returns
    -------
    torch.Tensor
        KL divergence, B, denoted as `kld`
    """
    kld_n = (
        torch.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 0.5
    )
    return kld_n.sum(-1)


class PipelineFactorVAE(Pipeline):
    """Pipeline to automate FactorVAE workflow"""

    def __init__(
        self,
        dataset: TensorDataset,
        partition: list[float],
        batch_size: int,
        loss_kwargs: dict[str, Any] | None = None,
        eval_kwargs: dict[str, Any] | None = None,
        shuffle_ds: bool = False,
        shuffle_dl: bool = True,
        loss_fn: Callable[..., torch.Tensor] = loss_fn_vae,
    ) -> None:
        super().__init__(
            loss_fn,
            dataset,
            partition,
            batch_size,
            loss_kwargs,
            eval_kwargs,
            shuffle_ds,
            shuffle_dl,
        )

    def calc_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: float,
        lmd: float,
    ) -> torch.Tensor:
        """Calculate the loss from the features and the label

        Parameters
        ----------
        model : nn.Module
            Model to be trained
        x : torch.Tensor
            Features, batch first
        y : torch.Tensor
            Labels, batch first
        gamma : float
            Gamma as regularization parameter
        lmd : float
            Lambda as regularization parameter

        Returns
        -------
        torch.Tensor
            Mean of the loss values across batches
        """
        out = model(x, y)
        loss = self.loss_fn(y, *out, gamma, lmd).mean(-1)
        return loss

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        test: bool,
        df: pd.DataFrame,
        top_pct: float,
    ) -> float:
        """Evaluate a model

        Parameters
        ----------
        model : nn.Module
            Model to be evaluated
        dataloader : DataLoader
            Dataloader (could be validation or testing)
        test : bool
            For testing or not
        df : pd.DataFrame
            Panel data
        top_pct : float
            Invest stocks with factor value in the top percentile, by default 0.1

        Returns
        -------
        float
            Out-of-sample Sharpe ratio
        """
        ds = next(iter(dataloader))
        x, _ = ds
        mu_y, _ = model.predict(x)

        # backtest
        len_eval = ds[1].shape[0]
        len_test = next(iter(self.dl_test))[1].shape[0]
        idx = pd.IndexSlice
        dates = df.index.get_level_values(0).unique()
        if test:
            df_eval = df.loc[idx[dates[-len_eval:], :]]
        else:
            df_eval = df.loc[idx[dates[-len_eval - len_test : -len_test], :]]
        df_eval["factor"] = mu_y.flatten().cpu().numpy()

        bt = Backtester("factor", cost=0.0, top_pct=top_pct).feed(df_eval).run()
        # bt.report(False)
        return bt.df_perf.loc["sharpe_ratio", "strat"]
