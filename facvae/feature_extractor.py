import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Feature extractor

    It extracts stocks hidden features `e` from the historical sequential
    characteristics `x`
    """

    def __init__(self, C: int, H: int, h_proj_size: int) -> None:
        """Initialization

        Parameters
        ----------
        T : int
            Size of time periods
        C : int
            Size of characteristics
        H : int
            Size of hidden features
        h_proj_size : int
            Size of hidden layer in projection layer
        """
        super(FeatureExtractor, self).__init__()
        self.C, self.H = C, H
        self.proj_layer = nn.Sequential(nn.Linear(C, h_proj_size), nn.LeakyReLU())
        self.gru = nn.GRU(input_size=h_proj_size, hidden_size=H, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden features

        Parameters
        ----------
        x : torch.Tensor
            Characteristics, B*N*T*C

        Returns
        -------
        torch.Tensor
            Hidden features, B*N*H, denoted as `e`
        """
        N, T = x.shape[1:3]  # N and T can be arbitrary
        x = x.reshape((-1, T, self.C))
        h_proj = self.proj_layer(x)
        _, h_T = self.gru(h_proj)
        e = h_T.reshape(-1, N, self.H)
        return e
