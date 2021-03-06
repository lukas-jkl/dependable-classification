from collections import OrderedDict
import torch.nn as nn


def create_model(num_hidden_layers: int, hidden_dim: int, dropout_p: float, in_features: int = 2, activation=nn.ReLU) \
                 -> nn.Module:
    """ Creates a simple model

    Args:
        num_hidden_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        dropout_p: probability to use for dropout
        in_features (int, optional): number of in features. Defaults to 2.
        activation: which activation function to use. Defaults to ReLU

    Returns:
        torch.nn.Module: model
    """
    layers = OrderedDict()
    prev_neurons = in_features

    for i in range(num_hidden_layers + 1):
        layers[str(i) + "_layer"] = nn.Linear(prev_neurons, hidden_dim)
        if i != num_hidden_layers:
            layers[str(i) + "_activation"] = activation()
            layers[str(i) + "_dropout"] = nn.Dropout(dropout_p)
        prev_neurons = hidden_dim
    layers["output_layer"] = nn.Linear(prev_neurons, 1)

    return nn.Sequential(layers)
