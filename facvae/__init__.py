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
    - having time period index with value of the type pd.Timestamp
    - sorted by time period index
    - having "ret" column if and only if "ret" means return and it's the last column
    - balanced if not otherwise specified
    - having no missing value if not otherwise specified
2) Letter case
    Lowercase letters are used to denote vectors or matrices (tensor); capital letters
    are used to denote scalars (constant)
"""

__version__ = "0.0.1"

__all__ = ["data", "pipeline"]


import torch
import torch.nn as nn

from .factor_decoder import FactorDecoder
from .factor_encoder import FactorEncoder
from .factor_predictor import FactorPredictor
from .feature_extractor import FeatureExtractor


class FactorVAE(nn.Module):
    """Factor VAE (Variational Auto-Encoder)

    It extracts effective factors from noisy market data. First, it obtain optimal
    factors by an encoder-decoder architecture with access to future data, and then
    train a factor predictor according a prior-posterior learning method, which extracts
    factors to approximate the optimal factors.

    Notation
    - Scalar (constant)
        - B: size of batch (arbitrary)
        - N: size of stocks (arbitrary)
        - T: size of time periods (arbitrary)
        - C: size of characteristics
        - H: size of hidden features
        - M: size of portfolios
        - K: size of factors
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
                Predicted cov matrix of stock returns, B*N*N, denoted as `Sigma_y`
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
        mu_y, Sigma_y = self.decoder(e, mu_post, sigma_post)
        mu_prior, sigma_prior = self.predictor(e)
        return mu_y, Sigma_y, mu_post, sigma_post, mu_prior, sigma_prior

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
                Predicted covariance matrix of stock returns, B*N, denoted as `Sigma_y`
        """
        with torch.no_grad():
            e = self.extractor(x)
            mu_prior, sigma_prior = self.predictor(e)
            mu_y, Sigma_y = self.decoder(e, mu_prior, sigma_prior)
        return mu_y, Sigma_y
