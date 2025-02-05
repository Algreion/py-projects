import torch
import torch.nn.functional as F
from random import shuffle
from matplotlib import pyplot as plt

# Main approach:
# - Convert words into N-dimensional vectors, initialized randomly.

#* Glossary
# Hyperparameter: Hidden layer with variable size which depends on how well they perform.
# Examples / Labels: Inputs outputs | input/example: emm -> [5,13,13], label: a -> [1]
# Embedding: 'cramming' data in lower-dimensional space, eg. 27 chars mapped to 2D vectors.
# Underfitting: Neural Network is too small, losses for dev and test splits are mostly equal.
# Overfitting: 'Overusing' a relatively small batch of data, loss -> 0. Model is just memorizing.
#   Fix: Split up training data into:
#       - Training split | 80%, actual training, optimizing parameters with gradient descent.
#       - Dev/validation split | 10%, optimizing settings eg. hyperparameters to find best
#       - Test split | 10%, Evaluate performance of model at the end (avoid overfitting)
# Mini-batch: Better to have 'approximate' gradient and make more steps (efficient)
#   than to calculate precise one but take much more time.
# Approach to training: Find good initial learning rate, train for a while, then decay

class Test:
    def __init__(self, file: str, blocks: int = 3, dims: int = 2, neurons: int = 100):
        """MLP?"""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read().splitlines()
        except:
            self.data = []
        self.blocks = blocks
        self.dims = dims
        self.lookup = dict(enumerate(['.']+list('abcdefghijklmnopqrstuvwxyz')))
        self.rlookup = dict([(v,k) for k,v in self.lookup.items()])
        self.W1 = torch.randn((self.dims*self.blocks, neurons), requires_grad=True) # First hidden layer
        self.b1 = torch.randn(neurons, requires_grad=True) # Biases of ^
        self.W2 = torch.randn((neurons, 27), requires_grad=True)
        self.b2 = torch.randn((27), requires_grad=True)
        self.C = torch.randn((27, self.dims), requires_grad=True) # Random initial N-dimensional vectors for chars
        self.params = [self.W1,self.b1,self.W2,self.b2,self.C]
        self.datasets = []
    
    def trainingset(self, size: int = 0, data: list = []) -> list:
        """Builds the full training set with n-size character inputs and expected outputs."""
        if not data: data = self.data
        X, Y = [], []
        for w in data[:size if size>0 else len(self.data)]:
            context = [0] * self.blocks
            for c in w+'.':
                index = self.rlookup[c]
                X.append(context)
                Y.append(index)
                context = context[1:] + [index]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return [X,Y]
    
    def splitdataset(self, override: bool = False) -> None:
        """Builds training set + dev set + test set. Stores them in self.datasets."""
        if override: self.datasets = []
        if self.datasets: return
        words = self.data[:]
        shuffle(words)
        n1 = int(0.8 * len(words))
        n2 = int(0.9 * len(words))
        Xtr, Ytr = self.trainingset(data=words[:n1])
        Xdev,Ydev = self.trainingset(data=words[n1:n2])
        Xtest,Ytest = self.trainingset(data=words[n2:])
        self.datasets.extend([Xtr, Ytr, Xdev, Ydev, Xtest, Ytest])

    def embedding(self, inputs: torch.tensor) -> torch.tensor:
        """Embed the inputs and return their logits."""
        embed = self.C[inputs]
        activation = torch.tanh(embed.view(-1, self.dims*self.blocks) @ self.W1 + self.b1)
        return activation @ self.W2 + self.b2

    def train(self, n: int = 1, info: bool = False, batch: int = 32, step: float = 0.1, full: bool = False) -> None:
        """Performs n training steps with mini-batches."""
        if full: X, Y = self.trainingset()
        else:
            if not self.datasets: self.splitdataset()
            X, Y = self.datasets[0], self.datasets[1]
        for _ in range(n):
            # Construct mini-batch
            index = torch.randint(0, X.shape[0], (batch,))

            # Forward pass
            logits = self.embedding(X[index])
            loss = F.cross_entropy(logits, Y[index])

            # Backward pass
            for p in self.params: p.grad = None # Zero-grad
            loss.backward()

            # Update
            for p in self.params:
                p.data -= step * p.grad
            if info: print(f"{_}. {loss.item():.4f}")
    
    def loss(self, dev: bool = True) -> torch.tensor:
        """Returns the total loss over the entire dataset / dev dataset."""
        if not dev: X, Y = self.trainingset()
        else:
            if not self.datasets: self.splitdataset()
            X, Y = self.datasets[2], self.datasets[3]
        logits = self.embedding(X)
        return F.cross_entropy(logits, Y)
    
    def generate(self) -> str:
        """Generate one word based on the training data."""
        out = ""
        index = 0
        while True:
            pass
        return out
    
    def generateN(self, n: int = 10) -> list:
        """Generate n words based on training data."""
        out = []
        for _ in range(n):
            out.append(self.generate())
        return out
    
    def size(self) -> int:
        """Returns the total size (number of parameters) of the model."""
        return sum(p.nelement() for p in self.params)
    
    def plot(self):
        """Shows the way the model maps the characters as vectors."""
        C = self.C
        if self.dims > 2:
            C = self.flatten()
        plt.figure(figsize=(8,8))
        plt.scatter(C[:,0].data, C[:,1].data, s = 200)
        for i in range(C.shape[0]):
            plt.text(C[i,0].item(), C[i,1].item(), self.lookup[i] ,ha="center",va="center",color="white")
        plt.grid('minor')
    
    def flatten(self, vectors = None) -> torch.tensor:
        """Reduces N-dimensional vectors to 2D (with SVD)"""
        if not vectors: vectors = self.C
        U, S, _ = torch.svd(vectors)
        return U[:, :2] @ torch.diag(S[:2])
