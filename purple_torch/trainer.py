import torch

from torch import nn


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        loss_function,
        optimizer,
        epochs,
        scheduler,
        scheduler_step_size: int = 1,
        stats_step_size: int = 1,
        to_cuda: bool = False,
    ):
        """Trains an instance of a NeuralNetwork.

        Args:
            model: purple_torch.NeuralNetwork
                An instance of a purple_torch.NeuralNetwork
            dataloader: torch.utils.data.dataloader.DataLoader
                A pytorch Dataloader.
            loss_function
                Loss function.
            optimizer: torch.optim
                A pytorch optimizer.
            epochs: int
                The number of epochs during training.
            scheduler: torch.optim.lr_scheduler
                A learning rate scheduler
            scheduler_step_size: int
                The number of passes during training before
                applying an update step to the scheduler. Default 1.
            stats_step_size: int
                The number of passes during training before updating
                the stats derived from training. Default 1.
            to_cude: bool
                Flag that moves to data to cuda. Default False (cpu).

        """
        self.device = 'cuda' if to_cuda else 'cpu'
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_size = scheduler_step_size
        self.stats_step_size = stats_step_size
        self.to_cuda = to_cuda

    def fit(self):
        """Fits a model"""
        pass

    def plot_stats(self):
        """Plots the loss, layer mean/std, and gradient norms."""
        pass

    def to_disk(self):
        """Saves the model to disk."""
        pass

    def forward_pass(self):
        """A single batch pass through the network. Does not backprop"""
        pass


