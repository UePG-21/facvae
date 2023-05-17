from typing import Callable

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.data import DataLoader


def loss_func(
    y: torch.Tensor,
    mu_y: torch.Tensor,
    Sigma_y: torch.Tensor,
    mu_post: torch.Tensor,
    sigma_post: torch.Tensor,
    mu_prior: torch.Tensor,
    sigma_prior: torch.Tensor,
    lmd: float = 1.0,
) -> torch.Tensor:
    """Loss function

    Parameters
    ----------
    y : torch.Tensor
        Stock returns, B*N
    mu_y : torch.Tensor
        Predicted mean vector of stock returns, B*N
    Sigma_y : torch.Tensor
        Predicted cov matrix of stock returns, B*N*N
    mu_post : torch.Tensor
        Means of posterior factor returns, B*K
    sigma_post : torch.Tensor
        Stds of posterior factor returns, B*K
    mu_prior : torch.Tensor
        Means of prior factor returns, B*K
    sigma_prior : torch.Tensor
        Stds of prior factor returns, B*K
    lmd : float, optional
        Lambda as regularization parameter, by default 1.0

    Returns
    -------
    torch.Tensor
        Loss value, denoted as `loss`
    """
    dist_y = MultivariateNormal(mu_y, Sigma_y)
    ll = dist_y.log_prob(y).sum(-1)
    kld = gaussian_kld(mu_post, mu_prior, sigma_post, sigma_prior)
    loss = -ll / y.shape[-1] + lmd * kld
    return loss.mean(-1)  # average loss across the batch


def gaussian_kld(
    mu1: torch.Tensor, mu2: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """Calculate KL divergence of two multivariate independent Gaussian distributions

    Parameters
    ----------
    mu1 : torch.Tensor
        Means of the first Gaussian, B*N
    mu2 : torch.Tensor
        Means of the second Gaussian, B*N
    sigma1 : torch.Tensor
        Stds of the first Gaussian, B*N
    sigma2 : torch.Tensor
        Stds of the second Gaussian, B*N

    Returns
    -------
    torch.Tensor
        KL divergence, B, denoted as `kld`
    """
    kld_n = (
        torch.log(sigma1 / sigma2)
        + (sigma1**2 + (mu1 - mu2) ** 2) / sigma2**2 / 2
        - 0.5
    )
    return kld_n.sum(-1)


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    learning_rate: float,
    epochs: int,
    max_grad: float | None = None,
    loss_func: Callable = loss_func,
    opt_family: torch.optim.Optimizer = torch.optim.Adam,
    verbose_freq: int = 10,
) -> None:
    """Train the model

    Parameters
    ----------
    model : nn.Module
        Model to be trained
    dataloader : DataLoader
        Training dataloader
    learning_rate : float
        Learning rate
    epochs : int
        Number of epoch to train the model
    max_grad : float, optional
        Max absolute value of the gradients for gradient clipping, by default None
    loss_func : Callable, optional
        Loss function, by default loss_func
    optimizer_family : torch.optim.Optimizer, optional
        Optimizer, by default torch.optim.Adam
    verbose_freq : int, optional
        Frequncy to report the loss, by default 10
    """
    # to cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # train
    optimizer: torch.optim.Optimizer = opt_family(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        print("=" * 16, f"Epoch {e}", "=" * 16)
        for b, (x, y) in enumerate(dataloader):
            # calculate loss
            out: tuple[torch.Tensor] = model(x, y)
            loss: torch.Tensor = loss_func(y, *out)
            loss.backward()
            # clip gradient
            if loss.item() > 1e9:
                optimizer.zero_grad()  # avoid getting inf gradients
            elif max_grad is not None:
                clip_grad_value_(model.parameters(), max_grad)
            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            # report loss
            if b % verbose_freq == 0:
                print(f"batch: {b}, loss: {loss.item()}")


def validate_model():
    # TODO: to be finished
    pass


def test_model(
    model: nn.Module, dataloader: DataLoader, loss_func: Callable = loss_func
) -> torch.Tensor:
    """Test the model

    Parameters
    ----------
    model : nn.Module
        Model to be tested
    dataloader : DataLoader
        Testing dataloader
    loss_func : Callable, optional
        Loss function, by default loss_func

    Returns
    -------
    torch.Tensor
        Loss value, denoted as `loss`
    """
    with torch.no_grad():
        # to cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss = torch.Tensor([0.0]).to(device)
        # calculate
        for x, y in dataloader:
            out: tuple[torch.Tensor] = model(x, y)
            loss += loss_func(y, *out)
    return loss / len(dataloader)
