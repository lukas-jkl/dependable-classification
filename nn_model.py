from collections import OrderedDict
import torch.nn as nn

def create_model(neurons, in_features=2, activation=nn.ReLU) -> nn.Module:
    """ Creates a simple model

    Args:
        neurons: list with number of neurons for each layer
        in_features (int, optional): number of in features. Defaults to 2.
        activation: which activation function to use. Defaults to ReLU

    Returns:
        torch.nn.Module: model
    """    
    layers = OrderedDict()
    prev_neurons = in_features
    
    for i, neuron in enumerate(neurons):
        layers[str(i) + "_layer"] = nn.Linear(prev_neurons, neuron)
        if i != len(neurons) - 1:
            layers[str(i) + "_activation"] = activation()
        prev_neurons = neuron
    layers["output_layer"] = nn.Linear(prev_neurons, 1)
    
    return nn.Sequential(layers)