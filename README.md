# MICROGRAD

A tiny autograd engine and neural network library, built from scratch in pure Python while following [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) lecture. It implements backpropagation over a dynamically built computation graph, and a small PyTorch-like neural network API on top of it.

## How it works

Every number is wrapped in a `Value` object. When you do math with `Value`s, they remember which operation created them and which values were their inputs — this forms a computation graph. Calling `.backward()` on the final result walks that graph in reverse (topological order) and applies the chain rule at every node, filling in `.grad` for every `Value`.

## Project structure

```
MICROGRAD/
├── engine.py          # Value class: the autograd engine
├── neuralnetwork.py   # Neuron, Layer, MLP built on top of Value
├── examples/
│   └── train.py       
└── requirements.txt
```

## Quick example

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + a.relu()
c.backward()

print(c.data)   # forward result: -4.0
print(a.grad)   # dc/da = -2.0  (b + relu gradient)
print(b.grad)   # dc/db =  2.0  (a)
```

## Training a neural net

```python
from neuralnetwork import MLP

model = MLP(3, [4, 4, 1])  
ypred = model([2.0, 3.0, -1.0])
```

Run the full training demo:

```
python examples/train.py
```

It trains a 41-parameter MLP with ReLU activations on 4 examples using mean squared error and plain gradient descent. Loss goes from ~3.6 to ~0.005 in 100 steps.

## Supported operations

`Value` supports: `+`, `-`, `*`, `/`, `**` (int/float powers), `exp()`, `tanh()`, `relu()`, `sigmoid()` — each with its backward (gradient) rule.

## Credits

Based on Andrej Karpathy's ["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0).
