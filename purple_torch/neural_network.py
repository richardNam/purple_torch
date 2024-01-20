import matplotlib.pyplot as plt
import torch

from torch import nn


class NeuralNetwork(nn.Module):
    """ A empty nn.Module class.

    Usage:

    import torch

    from torch import nn


    >>> autoencoder = NeuralNetwork(
    >>>     sequential_layers=nn.Sequential(
    >>>         nn.Conv2d(...),
    >>>         ...
    >>>         nn.Conv2d(...),
    >>>     ),
    >>>     initializer=nn.init.kaiming_normal_,
    >>>     track_gradients=True,
    >>> )

    >>> model = autoencoder.to('cuda')
    >>> loss_fun = nn.MSELoss()
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    >>> for batch, (X, y) in enumerate(torch.utils.data.dataloader.DataLoader):
    >>>     X, y = X.to('cuda'), y.to('cuda')

    >>>     pred = model.forward(X)
    >>>     loss = loss_fn(pred, X)
    >>>     loss.backward()

    >>>     model.append_gradients() # tracks gradients

    >>>     optimizer.step()
    >>>     optimizer.zero_grad()

    """
    def __init__(
        self,
        sequential_layers: nn.Sequential,
        initializer: nn.init = None,
        track_gradients: bool = False,

    ):
        """ Initializes a torch module

        Args:
            layers : nn.Sequential
                A sequence of torch layers
            initializer : nn.init
                A custom weight initializer, using the a set
                of nn.init functions. Default None.
            track_gradients : bool
                Flag to trigger tracking layer gradients during
                training. Default False.

        """
        super().__init__()
        self.layers = sequential_layers
        self.initializer = initializer
        self.track_gradients = track_gradients
        self.skip_list = (
            nn.BatchNorm2d,
            nn.Flatten,
            nn.ReLU,
            nn.Sigmoid,
            nn.UpsamplingNearest2d,

        )

        self.n_layers = len(self.layers)
        self.layer_names = [
            '{}-{}'.format(
                i+1, name.__class__.__name__
            ) for i, name in enumerate(self.layers)
        ]

        self.layer_means = [[] for _ in self.layers]
        self.layer_stds = [[] for _ in self.layers]
        self.layer_gradients = self._create_gradient_dict() if self.track_gradients else None

        if self.initializer is not None:
            self._initialize_weights()

    def forward(self, input_x: torch.Tensor):
        """Forward pass on the network, also stored means,
        standard deviations, and gradients of the layers.

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

    def append_gradients(self):
        """Appends the gradients used during training. Takes the norm of the
        gradients per layer. Should be called right after the loss is backprop
        and before the optimizer takes its next step. Needs to be called
        during training.

        Returns: None

        """
        if self.track_gradients:
            for i, layer_name in enumerate(self.layer_names):
                # The second condition assumes it is not met on the first pass,
                # i.e. not back-prop'd yet. TODO: throw a warning if it occurs more then once.
                if (layer_name in self.layer_gradients) and (self.layers[i].weight.grad is not None):
                    self.layer_gradients[layer_name].append(
                        # The to('cpu') can create a lot of overhead, TODO: speed up.
                        self.layers[i].weight.grad.norm().to('cpu').item()
                    )

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

    def plot_layer_gradients(self):
        """Plots the gradient"""

        layer_names = []
        for layer_name, gradients in self.layer_gradients.items():
            plt.plot(gradients)
            layer_names.append(layer_name)
        plt.hlines(0, 0, len(gradients), linestyles='dotted', color='grey')
        plt.legend(layer_names, loc=(1.04, 0))
        plt.xlabel('Pass Number')
        plt.ylabel('Gradient Norm')
        plt.title('Gradients')

    def _initialize_weights(self):
        """Applies an initializer to torch.nn layers. Checks and passes
        activiation and matrix transformation layers

        Returns: None
        """
        for layer in self.layers:
            if not isinstance(layer, self.skip_list):
                self.initializer(layer.weight)

    def _create_gradient_dict(self):
        """Creates a dictionary of that tracks the gradients of the layers
        over training batches.

        Returns: dict[str, list]

        """
        grad_tracker = {}
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, (self.skip_list)):
                layer_module = layer.__class__.__name__
                grad_tracker[f"{i+1}-{layer_module}"] = []

        return grad_tracker


    def plot_activation_stats(self):
        """Plots stats from activation functions"""
        pass

