# test.py
# Micrograd From Scratch
# Author: Zeeshan Ijaz
# Purpose: Test the automatic differentiation engine
#          and neural-network implementation

from engine import Value
from nn import Neuron, Layer, MLP


print("=" * 50)
print("MICROGRAD FROM SCRATCH")
print("=" * 50)


# =========================================================
# TEST 1 — BASIC VALUE
# =========================================================

print("\nTEST 1 — Value")

a = Value(2)
b = Value(3)

c = a + b

print("a =", a.data)
print("b =", b.data)
print("c =", c.data)
print("c operation =", c._op)

print("a is previous node:", a in c._prev)
print("b is previous node:", b in c._prev)


# =========================================================
# TEST 2 — MULTIPLICATION
# =========================================================

print("\nTEST 2 — Multiplication")

x = Value(2)
y = Value(3)

z = x * y

print("x =", x.data)
print("y =", y.data)
print("z =", z.data)


# =========================================================
# TEST 3 — BACKPROPAGATION
# =========================================================

print("\nTEST 3 — Backpropagation")

a = Value(2)
b = Value(3)

c = a * b

c.backward()

print("c =", c.data)
print("dc/da =", a.grad)
print("dc/db =", b.grad)


# =========================================================
# TEST 4 — MORE COMPLEX GRAPH
# =========================================================

print("\nTEST 4 — Computational Graph")

a = Value(2)
b = Value(3)

c = a + b
d = c * a

d.backward()

print("a =", a.data)
print("b =", b.data)
print("c =", c.data)
print("d =", d.data)

print("gradient of a =", a.grad)
print("gradient of b =", b.grad)


# =========================================================
# TEST 5 — NEURON
# =========================================================

print("\nTEST 5 — Neuron")

x = [
    Value(2.0),
    Value(3.0)
]

neuron = Neuron(2)

output = neuron(x)

print("Neuron output:", output)


# =========================================================
# TEST 6 — MLP
# =========================================================

print("\nTEST 6 — MLP")

model = MLP(3, [4, 4, 1])

inputs = [
    Value(2.0),
    Value(3.0),
    Value(-1.0)
]

prediction = model(inputs)

print("Model:", model)
print("Prediction:", prediction)


# =========================================================
# TEST 7 — BACKPROP THROUGH MLP
# =========================================================

print("\nTEST 7 — Neural Network Backpropagation")

target = Value(1.0)

prediction_value = prediction[0]

loss = (prediction_value - target) ** 2

loss.backward()

print("Prediction:", prediction_value.data)
print("Target:", target.data)
print("Loss:", loss.data)

print("\nNumber of parameters:", len(model.parameters()))

print("\nFirst 5 parameter gradients:")

for parameter in model.parameters()[:5]:
    print(parameter.grad)


print("\nAll tests completed.")