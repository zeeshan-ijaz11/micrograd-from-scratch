# nn.py
# Micrograd From Scratch
# Author: Zeeshan Ijaz
# Purpose: Simple neural-network components

import random
from engine import Value


class Neuron:

    def __init__(self, nin):
        """
        nin = number of inputs
        """

        # Random weights
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        # Bias
        self.b = Value(0)

    def __call__(self, x):

        # Weighted sum + bias
        activation = self.b

        for wi, xi in zip(self.w, x):
            activation = activation + wi * xi

        # Non-linear activation
        output = activation.tanh()

        return output

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer:

    def __init__(self, nin, nout):

        # Create nout neurons
        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x):

        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        return outputs

    def parameters(self):

        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

    def __repr__(self):
        return f"Layer({len(self.neurons)})"


class MLP:

    def __init__(self, nin, nouts):

        """
        Example:

        MLP(3, [4, 4, 1])

        means:

        3 inputs
        ↓
        4 neurons
        ↓
        4 neurons
        ↓
        1 output
        """

        sizes = [nin] + nouts

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):

        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def __repr__(self):
        return "MLP(" + ", ".join(
            str(len(layer.neurons))
            for layer in self.layers
        ) + ")"