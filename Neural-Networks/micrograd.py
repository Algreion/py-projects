from math import tanh,exp
import random

class Value:
    def __init__(self, data: float, _children=(), _op = ''):
        """Node for a data structure that links math operations together (Base for NN)"""
        self.data = data
        self.grad = 0 # Derivative (gradient) affecting the final output | Base case (for output) is 1.
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data})"
    def __str__(self):
        return str(self.data)
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # Closure, applied during backpropagation
            self.grad += 1.0 * out.grad # dx of a+b = 1
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other): # other + self
        return self + other
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self) + other
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad # dx of a*b = a
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other):
        return self*other
    def __pow__(self, number: float):
        assert isinstance(number, (int, float)), "Supports only ints and floats"
        out = Value(self.data**number, (self,), f'**{number}')
        def _backward():
            self.grad += number * self.data**(number-1) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * other**(-1) # a/b = a * (1/b)
    def __rtruediv__(self, other):
        return self**(-1) * other

    def exp(self):
        x = self.data
        out = Value(exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad # dx of e^x = e^x
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = tanh(x)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad # dx of tanh(x) = 1-tanh(x)^2
        out._backward = _backward
        return out
    
    def backward(self):
        """Backpropagation with previous nodes"""
        def topological_sort(root):
            """Adds a node to the list only after all its children have been processed"""
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(root)
            return topo
        
        self.grad = 1.0 # Base case
        for node in reversed(topological_sort(self)):
            node._backward()

#! Backpropagation
# We have a final output L, and want to find how everything impacts it (derivatives)
# Derivative of previous nodes (neurons!) is self-explanatory.
# Going back one layer, we apply the chain rule, multiplying the derivatives together. | LOCAL GRADIENT!
# ^ If dx=3 and dy=2, where x->y->L, then a tiny nudge to x causes 3 times the nudge in y, which in turn causes 2 times the nudge in L. Hence 3*2 = 6 for the impact x has to L.
# This way, we can recursively go back and find how each neuron impacts the final output!
# Example: x -> y -> L
# dL/dy = 3, dy/dx = 2, then dL/dx = (dL/dy)*(dy/dx)=3*2 = 6 | You know how all nodes impact the final one.
#* _backward() is a closure to apply local gradient onto previous nodes
#* In this case, it is applied after a topological sort
#* We must add the derivatives to the gradient since we have multiple variables.

#* Nudging all values in the direction of their gradient, we increase the final output


class Neuron:
    def __init__(self, nin: int):
        """Single NN Neuron. nin = Number of expected inputs"""
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # Random weight for each input
        self.b = Value(random.uniform(-1,1)) # Random bias for neuron
    def __repr__(self):
        return f"Neuron(inputs={len(self.w)},weights=[{",".join(str(round(w.data,2)) for w in self.w)}],bias={self.b.data:.2f})"
    def __str__(self):
        return f"N({len(self.w)})"
    def __call__(self, input: list):
        """Get the neuron's activation given a group of inputs"""
        # w * x + b
        activation = sum((w_i*x_i for w_i,x_i in zip(self.w, input)), self.b)
        return activation.tanh()

class Layer:
    def __init__(self, nin: int, nout: int):
        """Layer of NN Neurons.
        nin = Number of expected inputs.
        nout = Number of neurons in the layer"""
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __getitem__(self, index: int):
        return self.neurons[index]
    def __repr__(self):
        return f"Layer(inputs={len(self.neurons[0].w)},neurons={len(self.neurons)})"
    def __str__(self):
        return f"L({len(self.neurons)})"
    def __iter__(self):
        return iter(self.neurons)
    def __call__(self, input):
        """Returns the activations for all neurons in the layer."""
        out = [n(input) for n in self.neurons]
        return out[0] if len(out) == 1 else out

class MLP:
    def __init__(self, nin: int, nouts: list):
        """Multi-layer Perceptron.
        nin is number of inputs.
        nouts is list for number of neurons per layer"""
        size = [nin] + nouts
        self.layers = [Layer(size[i],size[i+1]) for i in range(len(nouts))]
    def __call__(self, input):
        for layer in self.layers:
            input = layer(input) # Iteratively update the result for every layer
        return input # Result (for final layer, the output)
    def __getitem__(self, index: int):
        return self.layers[index]
    def __repr__(self):
        return f"MLP(size={len(self.layers)},layers=[{", ".join(str(len(l.neurons)) for l in self.layers)}])"
    def __str__(self):
        res = ""
        res += f"[{" ".join("I" for _ in range(len(self[0][0].w)))}]\n"
        res += "\n".join(f"[{" ".join("N" for _ in l.neurons)}]" for l in self.layers[:-1])
        res += f"\n[{" ".join("O" for _ in self[-1].neurons)}]"
        return res


def forwardpass():
    # Start with 2 inputs (-2, 3)
    # For each Neuron (first):
    #   Initiate 2 random weights, and a random bias
    #   Activation = tanh(bias + sum(weight*input for w,i in Inputs))
    # For each Layer (first):
    #   Initiate 3 neurons (nout) each with 2 inputs (nin)
    #   Activation: Calls activation for each neuron and returns it
    # For the MLP:
    #   Initiate 3 layers, with input i and neurons i+1
    x = [-2, 3]
    mlp = MLP(2, [3,2,1])
    print(mlp)
    return mlp(x)
