import math

class Value:
    def __init__(self, data: float, _children=(), _op = ''):
        """Node for data structure that links operations together (Base for NN)"""
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
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # Closure, applied during backpropagation
            self.grad = 1.0 * out.grad # dx of a+b = 1
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad # dx of a*b = a
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad = (1 - t**2) * out.grad # dx of tanh(a) = 1-tanh(a)^2
        out._backward = _backward
        return out
    
    def backward(self):
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
        
        self.grad = 1.0
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

#* Nudging all values in the direction of their gradient, we increase the final output



def example_nn():
    # Input neurons x1,x2
    x1,x2 = Value(2.0), Value(0.0)
    # Weights w1, w2
    w1,w2 = Value(-3.0), Value(1.0)
    # Bias of the neuron
    b = Value(6.9)

    x1w1, x2w2 = x1*w1, x2*w2 # Individual edges
    totalsum = x1w1 + x2w2 # What arrives to the neuron
    n = totalsum + b # Value of the neuron
    output = n.tanh() # Final output (true value) of the neuron, activation function
    return output