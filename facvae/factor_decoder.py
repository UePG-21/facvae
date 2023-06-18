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

    def forward(self, e: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor]:
        """Get distribution parameters of predicted stock returns (with sampled `z`)

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H
        z : torch.Tensor
            Factor returns (could be `z_post` or `z_prior`), B*K

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Predicted mean vector of stock returns, B*N, denoted as `mu_y`
            torch.Tensor
                Predicted std vector of stock returns, B*N, denoted as `sigma_y`
        """
        mu_alpha, sigma_alpha = self.alpha_layer(e)
        beta = self.beta_layer(e)
        mu_y = mu_alpha + torch.einsum("bnk, bk -> bn", beta, z)
        sigma_y = sigma_alpha
        return mu_y, sigma_y

    def predict(
        self, e: torch.Tensor, mu_z: torch.Tensor, sigma_z: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Get distribution parameters of predicted stock returns (without sampled `z`)

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
        Sigma_z = torch.diag_embed(sigma_z**2)
        Sigma_y = torch.bmm(torch.bmm(beta, Sigma_z), beta.permute(0, 2, 1))
        Sigma_y += torch.diag_embed(sigma_alpha**2)
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
        self.mean_layer = nn.Linear(h_alpha_size, 1)
        self.std_layer = nn.Sequential(nn.Linear(h_alpha_size, 1), nn.Softplus())

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
                Means of idiosyncratic returns, B*N, denoted as `mu_alpha`
            torch.Tensor
                Stds of idiosyncratic returns, B*N, denoted as `sigma_alpha`
        """
        h_alpha = self.h_layer(e)
        mu_alpha = self.mean_layer(h_alpha).squeeze(-1)
        sigma_alpha = self.std_layer(h_alpha).squeeze(-1)
        return mu_alpha, sigma_alpha


class BetaLayer(nn.Module):
    """Beta layer

    It calculates factor exposures `beta` from hidden feautres `e`
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
