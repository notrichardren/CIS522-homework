#%%

import torch
from typing import Callable


#%%

class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int, # input-dim
        hidden_size: int, 
        num_classes: int, # output-dim
        hidden_count: int = 1, # number of hidden dimensions
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super().__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(hidden_count):
            if i == 0:
                # layerList.append(torch.nn.Linear(input_size, hidden_size))
                self.layers += [torch.nn.Linear(input_size, hidden_size)]
                self.layers += [activation()]
            elif i == hidden_count - 1:
                self.layers += [torch.nn.Linear(hidden_size, num_classes)]
            else:
                self.layers += [torch.nn.Linear(hidden_size, hidden_size)]
                self.layers += [activation()]    

        for layer in list(self.layers):
            if isinstance(layer, torch.nn.Linear):
                initializer(layer.weight.data)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for function in self.layers:
            x = function(x)
        return x
        
# %%
