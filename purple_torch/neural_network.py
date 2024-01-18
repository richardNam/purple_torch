import matplotlib.pyplot as plt

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers: dict):
        """ Initializes a torch module
        Args:
            layers : dict[str: nn.*]
                A dictionary of torch nn layers, as values, and layer
                descriptions as the keys.
        
        """
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.layer_keys = list(self.layers.keys())
        self.layer_means = {layer_key: [] for layer_key in self.layer_keys}
        self.layer_stds = {layer_key: [] for layer_key in self.layer_keys}
        
        
    def forward(self, x):
        """A forward pass on the netowrk. Stores, the layer means and stds.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Output after a full pass in the network.
        """
        for layer_key, layer_value in self.layers.items():
            x = self.layers[layer_key](x)
            # each element represents a pass
            self.layer_means[layer_key].append(x.mean().item())
            self.layer_stds[layer_key].append(x.std().item())
            
        return x
    
    def plot_layer_means(self):
        for layer_key in self.layer_keys:
            plt.plot(self.layer_means.get(layer_key))
        plt.legend(self.layer_keys, loc=(1.04, 0))
        plt.xlabel('Pass Number')
        plt.ylabel('Mean')
        plt.title('Means')
        
        
    def plot_layer_stds(self):
        for layer_key in self.layer_keys:
            plt.plot(self.layer_stds.get(layer_key))
        plt.legend(self.layer_keys, loc=(1.04, 0))
        plt.xlabel('Pass Number')
        plt.ylabel('Std')
        plt.title('Standard Deviations')
