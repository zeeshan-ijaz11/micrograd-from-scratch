# engine.py
# Micrograd From Scratch
# Author: Zeeshan Ijaz
# Purpose: Educational automatic differentiation engine

import math


class Value:
    """
    A scalar value that stores:
    - its numerical data
    - its gradient
    - previous nodes in the computational graph
    - the operation that created it
    - a function used for backpropagation
    """

    def __init__(self, data, _children=(), _op='', label=''):
        # Actual numerical value
        self.data = data

        # Gradient of the final output with respect to this value
        self.grad = 0.0

        # Previous Value objects that created this Value
        self._prev = set(_children)

        # Operation that created this Value
        self._op = _op

        # Optional label for easier debugging
        self.label = label

        # Function that calculates local gradients
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    # ---------------------------------------------------------
    # ADDITION
    # ---------------------------------------------------------

    def __add__(self, other):

        # Allow normal Python numbers such as:
        # Value(2) + 3
        other = other if isinstance(other, Value) else Value(other)

        # Calculate the forward-pass result
        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        # Define how gradients flow backward
        def _backward():

            # d(self + other) / d(self) = 1
            self.grad += 1.0 * out.grad

            # d(self + other) / d(other) = 1
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # ---------------------------------------------------------
    # MULTIPLICATION
    # ---------------------------------------------------------

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        # Forward pass
        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        # Backward pass
        def _backward():

            # d(self * other) / d(self) = other
            self.grad += other.data * out.grad

            # d(self * other) / d(other) = self
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # ---------------------------------------------------------
    # POWER
    # ---------------------------------------------------------

    def __pow__(self, other):

        assert isinstance(other, (int, float)), \
            "Power must be an integer or float."

        out = Value(
            self.data ** other,
            (self,),
            f'**{other}'
        )

        def _backward():

            # d(x^n)/dx = n*x^(n-1)
            self.grad += (
                other
                * self.data ** (other - 1)
                * out.grad
            )

        out._backward = _backward

        return out

    # ---------------------------------------------------------
    # NEGATIVE
    # ---------------------------------------------------------

    def __neg__(self):
        return self * -1

    # ---------------------------------------------------------
    # SUBTRACTION
    # ---------------------------------------------------------

    def __sub__(self, other):
        return self + (-other)

    # ---------------------------------------------------------
    # DIVISION
    # ---------------------------------------------------------

    def __truediv__(self, other):
        return self * other ** -1

    # ---------------------------------------------------------
    # RIGHT-SIDE OPERATIONS
    # ---------------------------------------------------------

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    # ---------------------------------------------------------
    # TANH ACTIVATION
    # ---------------------------------------------------------

    def tanh(self):

        x = self.data

        # tanh(x)
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(
            t,
            (self,),
            'tanh'
        )

        def _backward():

            # derivative of tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    # ---------------------------------------------------------
    # BACKPROPAGATION
    # ---------------------------------------------------------

    def backward(self):

        # Store all nodes in topological order
        topo = []

        # Keep track of visited nodes
        visited = set()

        def build_topo(v):

            if v not in visited:

                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        # Build graph starting from the output
        build_topo(self)

        # Final output derivative with respect to itself = 1
        self.grad = 1.0

        # Move backward through the graph
        for node in reversed(topo):
            node._backward()