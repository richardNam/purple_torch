import matplotlib.pyplot as plt
import torch

from torch import nn


class NeuralNetwork(nn.Module):
    """ A empty nn.Module class. """

    def __init__(
            self,
            sequential_layers: nn.Sequential,
            initializer: nn.init = None
    ):
        """ Initializes a torch module

        Args:
            layers : nn.Sequential
                A sequence of torch layers
            initializer : nn.init
                A custom weight initializer, using the a set
                of nn.init functions. Default None.
        """
        super().__init__()
        self.layers = sequential_layers
        self.initializer = initializer
        self.n_layers = len(self.layers)
        self.layer_names = [
            '{} - {}'.format(
                i+1, name.__class__.__name__
            ) for i, name in enumerate(self.layers)
        ]
        self.layer_means = [[] for _ in self.layers]
        self.layer_stds = [[] for _ in self.layers]

    def forward(self, input_x: torch.Tensor):
        """Forward pass on the network, also stored means
        and standard deviations of the laters.

        Args:
            x: Torch.tensor

        Returns: Torch.tensor
        """
        for i, _ in enumerate(self.layers):
            input_x = self.layers[i](input_x)
            # each element represents a pass
            self.layer_means[i].append(input_x.mean().item())
            self.layer_stds[i].append(input_x.std().item())

        return input_x

    def plot_layer_means(self):
        """Returns a plot of the means for each layers for
        each pass of the data.

        Returns: None

        """
        for mean in self.layer_means:
            plt.plot(mean)
        plt.legend(self.layer_names, loc=(1.04, 0))
        plt.xlabel('Pass Number')
        plt.ylabel('Mean')
        plt.title('Means')

    def plot_layer_stds(self):
        """Returns a plot of standard deviations for each
        layer for each pass of the data.

        Returns: None

        """
        for std in self.layer_stds:
            plt.plot(std)
        plt.legend(self.layer_names, loc=(1.04, 0))
        plt.xlabel('Pass Number')
        plt.ylabel('Std')
        plt.title('Standard Deviations')

    def _initialize_weights(self):
        """Applies an initializer to torch.nn layers. Checks and passes
        activiation and matrix transformation layers

        Returns:
            None
        """
        skip_list = (
            nn.ReLU,
            nn.Sigmoid,
            nn.Flatten,
            nn.UpsamplingNearest2d,
        )
        for layer in self.layers:
            if not isinstance(layer, skip_list):
                self.initializer(layer.weight)
