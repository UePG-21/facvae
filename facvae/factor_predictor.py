import torch
import torch.nn as nn


class FactorPredictor(nn.Module):
    """Factor predictor

    It extracts prior factor returns `z_prior`, a random vector following the
    independent Gaussian distribution, which can be described by the mean `mu_prior` and
    the standard deviation `sigma_prior`, from hidden features `e`.
    """

    def __init__(self, H: int, K: int, h_prior_size: int) -> None:
        """Initialization

        Parameters
        ----------
        H : int
            Size of hidden features
        K : int
            Size of factors
        h_prior_size : int
            Size of hidden layer in prior layer
        """
        super(FactorPredictor, self).__init__()
        self.attention = MultiheadGlobalAttention(H, K, H)
        self.h_layer = nn.Linear(H, h_prior_size)
        self.mu_layer = nn.Linear(h_prior_size, 1)
        self.sigma_layer = nn.Sequential(nn.Linear(h_prior_size, 1), nn.Softplus())

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor]:
        """Get distribution parameters of prior factor returns

        Parameters
        ----------
        e : torch.Tensor
            Hidden features, B*N*H

        Returns
        -------
        tuple[torch.Tensor]
            torch.Tensor
                Means of prior factor returns, B*K, denoted as `mu_prior`
            torch.Tensor
                Stds of prior factor returns, B*K, denoted as `sigma_prior`
        """
        h_att = self.attention(e)
        h_prior = self.h_layer(h_att)
        mu_prior = self.mu_layer(h_prior).squeeze(-1)
        sigma_prior = self.sigma_layer(h_prior).squeeze(-1)
        return mu_prior, sigma_prior


class MultiheadGlobalAttention(nn.Module):
    """Multi-head global attention (a special implement)

    From e in R^{token_size(N)*embed_dim} to h in R^{num_heads*value_dim}:
    k_n = W_key @ e_n, v_n = W_value @ e_n
    s_n = q @ k_n^T / ||q||_2 * ||k_n||_2
    a_n = s_n / sum_{m=1}^{N}{s_m}
    h = sum_{n=1}^{N}{a_n * v_n}
    """

    def __init__(self, embed_dim: int, num_heads: int, value_dim: int) -> None:
        """Initialization

        Parameters
        ----------
        embed_dim : int
            Embedding dimension
        num_heads : int
            Number of heads
        value_dim : int
            Value dimension
        """
        super(MultiheadGlobalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.q = nn.Parameter(torch.rand(num_heads, embed_dim))
        self.k_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_layer = nn.Linear(embed_dim, value_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get multi-head attention values

        Parameters
        ----------
        x : torch.Tensor
            Sequence of embedding vector, B*N*embed_dim, N can be arbitrary

        Returns
        -------
        torch.Tensor
            Multi-head attention values, num_heads*value_dim, denoted as `h`
        """
        if x.shape[-1] != self.embed_dim:
            raise Exception("input shape incorrect")
        k, v = self.k_layer(x), self.v_layer(x)
        q_norm, k_norm = torch.norm(self.q, dim=-1), torch.norm(k, dim=-1)
        s = torch.matmul(k, self.q.T) / q_norm / k_norm.unsqueeze(-1)
        a = s / torch.sum(s, dim=-1).unsqueeze(-1)  # B*num_heads*token_size
        h = torch.einsum("bnt, bnv -> btv", a, v)
        return h


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from feature_extractor import FeatureExtractor

    dir_main = "E:/Others/Programming/py_vscode/projects/signal_mixing/"
    dir_code = dir_main + "code/"
    dir_data = dir_main + "data/"
    dir_config = dir_main + "config/"
    dir_result = dir_main + "result/"

    H = 16
    N = 74
    T = 20
    C = 10
    M = 18
    K = 8
    h_proj_size = 6
    h_alpha_size = 6
    h_prior_size = 6

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

    fp = FactorPredictor(H, K, h_prior_size)
    mu_prior, sigma_prior = fp(e)
    print(mu_prior.shape)
    print(mu_prior)
    print(sigma_prior.shape)
    print(sigma_prior)
