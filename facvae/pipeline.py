import os
import random
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.data import DataLoader, Subset, TensorDataset


def set_seeds(seed: int) -> None:
    """Set random seeds

    Parameters
    ----------
    seed : int
        Random seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Pipeline(ABC):
    """Machine learning pipeline"""

    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        dataset: TensorDataset,
        partition: list[float],
        batch_size: int,
        loss_kwargs: dict[str, Any] | None = None,
        eval_kwargs: dict[str, Any] | None = None,
        shuffle_ds: bool = False,
        shuffle_dl: bool = True,
    ) -> None:
        """Initialization

        Parameters
        ----------
        loss_fn : Callable[..., torch.Tensor]
            Loss function
        dataset : TensorDataset
            Full dataset
        partition : list[float]
            Percentages of training, validation, and testing dataset
        batch_size : int
            How many samples per batch to load in training step
        loss_kwargs : dict[str, Any] | None
            Keyword arguments for `calc_loss()`, by default None
        eval_kwargs : dict[str, Any] | None
            Keyword arguments for `evaluate()`, by default None
        shuffle_ds : bool, optional
            Shuffle the full dataset before spliting or not, by default False
        shuffle_dl : bool, optional
            Set to True to have the data reshuffled at every epoch, by default False
        """
        self.loss_fn = loss_fn
        ds_train, ds_valid, ds_test = self._tvt_split(dataset, partition, shuffle_ds)
        self.dl_train = DataLoader(ds_train, batch_size, shuffle_dl)
        self.dl_valid = DataLoader(ds_valid, len(ds_valid)) if len(ds_valid) else None
        self.dl_test = DataLoader(ds_test, len(ds_test))
        self.loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        self.eval_kwargs = eval_kwargs if eval_kwargs is not None else {}

    def _tvt_split(
        self, dataset: TensorDataset, partition: list[float], shuffle: bool = False
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
        if abs(sum(partition) - 1.0) > 1e-9 or len(partition) != 3:
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

    @abstractmethod
    def calc_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, **loss_kwargs: Any
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
        loss_kwargs : dict[str, Any]
            Keyword arguments

        Returns
        -------
        torch.Tensor
            Mean of the loss values across batches
        """
        pass

    @abstractmethod
    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, test: bool, **eval_kwargs: Any
    ) -> Any:
        """Evaluate a model

        Parameters
        ----------
        model : nn.Module
            Model to be evaluated
        dataloader : DataLoader
            Dataloader (could be validation or testing)
        test : bool
            For testing or not
        eval_kwargs : dict[str, Any]
            Keyword arguments

        Returns
        -------
        Any
            Evaluation result
        """
        pass

    def train(
        self,
        model: nn.Module,
        learning_rate: float,
        epochs: int,
        max_grad: float | None = None,
        optim_algo: optim.Optimizer = optim.Adam,
        verbose_freq: int | None = 10,
    ) -> None:
        """Train a model

        Parameters
        ----------
        model : nn.Module
            Model to be trained
        learning_rate : float
            Learning rate
        epochs : int
            Number of epoch to train the model
        max_grad : float, optional
            Max absolute value of the gradients for gradient clipping, by default None
        optim_algo : torch.optim.Optimizer, optional
            Optimization algorithm, by default optim.Adam
        verbose_freq : int | None, optional
            Frequncy to report the loss, by default 10
        """
        # to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # train
        optimizer = optim_algo(model.parameters(), lr=learning_rate)
        for e in range(epochs):
            print("=" * 16, f"Epoch {e}", "=" * 16)
            for b, (x, y) in enumerate(self.dl_train):
                # calculate loss
                loss = self.calc_loss(model, x, y, **self.loss_kwargs)
                # back propagate
                optimizer.zero_grad()
                loss.backward()
                # clip gradient
                if max_grad is not None:
                    clip_grad_value_(model.parameters(), max_grad)
                # update parameters
                optimizer.step()
                # report loss
                if verbose_freq is not None and b % verbose_freq == 0:
                    print(f"batch: {b}, loss: {loss.item()}")

    @torch.no_grad()
    def validate(self, model: nn.Module) -> Any:
        """Validate a model

        Parameters
        ----------
        model : nn.Module
            Model to be validated

        Returns
        -------
        Any
            Validation result
        """
        # to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # validate
        return self.evaluate(model, self.dl_valid, False, **self.eval_kwargs)

    @torch.no_grad()
    def test(self, model: nn.Module) -> Any:
        """Test a model

        Parameters
        ----------
        model : nn.Module
            Model to be tested

        Returns
        -------
        Any
            Testing result
        """
        # to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # test
        return self.evaluate(model, self.dl_test, True, **self.eval_kwargs)
