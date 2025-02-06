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


class Ngram:
    def __init__(self, file: str, context: int = 3, dims: int = 2, neurons: int = 100):
        """Character n-gram: Neural Network to predict next character based on training data."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read().splitlines()
        except:
            self.data = []
        self.context = context
        self.dims = dims
        self.lookup = dict(enumerate(['.']+list('abcdefghijklmnopqrstuvwxyz')))
        self.rlookup = dict([(v,k) for k,v in self.lookup.items()])
        self.W1 = torch.randn((self.dims*self.context, neurons), requires_grad=True) # First hidden layer
        self.b1 = torch.randn(neurons, requires_grad=True) # Biases of ^
        self.W2 = torch.randn((neurons, 27), requires_grad=True)
        self.b2 = torch.randn((27), requires_grad=True)
        self.C = torch.randn((27, self.dims), requires_grad=True) # Random initial N-dimensional vectors for chars
        self.params = [self.W1,self.b1,self.W2,self.b2,self.C]
        self.datasets = []
    def __repr__(self):
        return f"Ngram(sample={self.generate()},context={self.context},size={self.size()})"
    def __str__(self):
        """Used to validate saved parameters."""
        return f"NGRAM[{",".join(map(lambda x: str(x),[self.context,self.dims,self.size()]))}]"
    def __call__(self, word: str = ''):
        return self.generate(word)
    
    def trainingset(self, size: int = 0, data: list = []) -> list:
        """Builds the full training set with n-size character inputs and expected outputs."""
        if not data: data = self.data
        X, Y = [], []
        for w in data[:size if size>0 else len(self.data)]:
            context = [0] * self.context
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

    #* Main process:
    # Get inputs (vectors of length 'context') and their expected outputs | ex: [1 2 0] -> [3] ("ab."->"abc")
    #? a = n.trainingset(1)
    # Lookup each character into self.C lookup table, which turns them into N-dimensional vectors | ex: 0 -> [2.14 -1.31]
    # Then we have many context-sized tensors with n-dimensional vectors for chars inside.
    #? b = n.C[a] | See the lists of n-dim vectors
    # We view the embedding with dims*context columns. Each row has our actual inputs
    #? b = b.view(-1, n.dims * n.context)
    # Matrix multiplication with weights 1, then + 100-D bias vector 1. This returns the activation of all (100) neurons in the first hidden layer as a 100-D vector.
    # tanh non-linear function on ^
    # Matrix multiplication with weights 2, + 27-D bias vector 2, returning a 27-D vector, the logits
    #? logits = n.embedding(a)

    def embedding(self, inputs: torch.tensor) -> torch.tensor:
        """Embed the inputs and return their logits."""
        embed = self.C[inputs] # Chars' corresponding N-dimensional vectors
        activation = torch.tanh(embed.view(-1, self.dims*self.context) @ self.W1 + self.b1)
        return activation @ self.W2 + self.b2

    def train(self, n: int = 1, info: bool = False, batches: int = 32, step: float = 0.1, trainset: bool = True) -> None:
        """Performs n training steps with mini-batches.
        info: Show steps and loss.
        batches: How many training examples are given at a time.
        step: Gradient descent. 0.1 early on -> 0.01 near the end of training.
        trainset: By default only train on the training-set. If false, trains on the whole dataset."""
        if not trainset: X, Y = self.trainingset()
        else:
            if not self.datasets: self.splitdataset()
            X, Y = self.datasets[0], self.datasets[1]
        if batches > len(X) or batches == 0: batches = len(X)//10
        for _ in range(n):
            # Construct mini-batch
            index = torch.randint(0, X.shape[0], (batches,))

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
    
    def generate(self, word: str = '', lenlimit: int = 100) -> str:
        """Generate one word based on the training data."""
        out = [self.rlookup[c] for c in word]
        if word: 
            if self.context < len(word):
                context = [self.rlookup[c] for c in word][-self.context:]
            else:
                context = [0] * (self.context-len(word)) + [self.rlookup[c] for c in word]
        else: context = [0] * self.context # '...'
        while True:
            probs = F.softmax(self.embedding(torch.tensor(context)),dim=1)
            index = torch.multinomial(probs, num_samples=1, replacement=True).item()
            if index == 0 or len(out) >= lenlimit: break
            context = context[1:] + [index] # Shift context window
            out.append(index)
        return ''.join(self.lookup[i] for i in out)
    
    def generateN(self, n: int = 10, word: str = '') -> list:
        """Generate n words based on training data."""
        out = []
        for _ in range(n):
            out.append(self.generate(word))
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
    
    def save(self, file: str, validation: bool = True):
        """Save the model's parameters on given file."""
        with open(file, 'w') as f:
            if validation: f.write(str(self)+'\n')
            for m in self.params:
                for p in m.view(-1): 
                    f.write(str(p.item())+'\n')

    def load(self, file: str, validation: bool = True):
        """Load parameters from a file. Must be of the same type as original."""
        check = False
        if validation:
            with open(file, 'r') as f:
                first = f.readline().strip()
                if first.startswith("NGRAM["):
                    print(first)
                    check = True
                    if first != str(self):
                        context,dims,c = map(int,first.removeprefix('NGRAM[').removesuffix(']').split(','))
                        neurons = int((c-27-dims*27)/(28+dims*context))
                        print(f"N-gram doesn't match structure: {context=} | {dims=} | {neurons=}.")
                        return
        with open(file,'r') as f:
            if check: next(f)
            for m in self.params:
                data = [float(f.readline()) for _ in range(m.numel())]
                m.data.copy_(torch.tensor(data, dtype=m.dtype).view(m.shape))
                m.grad = None

