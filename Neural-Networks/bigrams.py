import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

#! Bigram Language Model
# Considers 2 characters at a time
# Given 1 char, predict the next in the sequence. (a_ -> ab)
# We include a boundary character '/' (usually <S> and <E>, but this works fine)

#* The Bigram class is based on a NN, the HeuristicBigram class is based on pure heuristics.
#* Note that a very large input data may be required (eg. 32k+ English words/names) for decent results.

class Bigram:
    def __init__(self, file: str):
        """Trainable Neural Network model that predicts the next character based on given data. Fundamentally the same as micrograd's approach, using pytorch."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read().splitlines()
        except:
            self.data = []
        self.chars = ['/']+list('abcdefghijklmnopqrstuvwxyz')
        self.lookup = {s: i for i,s in enumerate(self.chars)}
        self.rlookup = {i: s for i,s in enumerate(self.chars)}

        self.weights = torch.randn((27,27), requires_grad=True) # 27 neurons with 27 inputs
    def __repr__(self):
        sample = self.generate()
        parameters = self.weights.nelement()
        return f"Bigram_Model({sample=},{parameters=})"
    def __call__(self, char: str = ''):
        return self.generate(char)
    
    def trainingset(self, n: int = 0) -> list:
        """Returns a list containing the inputs and expected outputs from the data."""
        if n == 0: n = len(self.data)
        inputs, outputs = [], []
        for w in self.data[:n]:
            chars = ['/'] + list(w) + ['/']
            for c1, c2 in zip(chars, chars[1:]):
                i1, i2 = self.lookup[c1], self.lookup[c2]
                inputs.append(i1)
                outputs.append(i2)
        return [torch.tensor(inputs), torch.tensor(outputs)]
    
    def train(self, n: int = 1, info: bool = False, step: float = 50.0):
        """Train the model n times based on data. A high step(10-100) works well."""
        for i in range(n):
            inp, out = self.trainingset()
            # Forward pass
            encoded = F.one_hot(inp, num_classes=27).float()
            logits = encoded @ self.weights # Matrix multiplication w input & neurons
            counts = logits.exp() # Softmax
            probs = counts / counts.sum(1, keepdim=True) # Normalized | Tensor of 27x27 probabilities (for the next character)
            p = probs[torch.arange(inp.nelement()), out] # Access the expected chars' probabilities
            cost = -p.log().mean() # + 0.01*(self.weights**2).mean() |< Regularization, 'smooths' out the probabilities | Actual cost for the model (NLL)

            # Backward Pass
            self.weights.grad = None # Zero-grad
            cost.backward() # Calculates gradients for all weights

            # Update
            self.weights.data -= step*self.weights.grad
            if info: print(f"{i}. {cost.item():.4f}")
    
    def activate(self, char: int) -> int:
        """Predicts the next character in bigram. Returns the corresponding index."""
        if type(char) == str: char = self.lookup[char]
        encoded = F.one_hot(torch.tensor(char), num_classes=27).float()
        logits = encoded @ self.weights
        counts = logits.exp()
        probs = counts / counts.sum()
        return torch.multinomial(probs, num_samples=1, replacement=True).item()
    
    def generate(self, char: str = '') -> str:
        """Generates and returns a string constructed from the data's training."""
        output = char
        index = self.lookup[char[-1]] if char else 0 # Begin the word (start token)
        while True:
            index = self.activate(index)
            if index == 0: # End token
                break
            output += self.rlookup[index]

        return output
    
    def generateN(self, n: int = 10) -> list:
        """Generates N strings."""
        result = []
        for _ in range(n):
            result.append(self.generate())
        return result



class HeuristicBigram:
    def __init__(self, file: str):
        """Model to predict next character given some training data. Inherently limited due to its purely heuristic approach (without parameters)."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read().splitlines()
        except:
            self.data = []
        self.chars = ['/']+list('abcdefghijklmnopqrstuvwxyz')
        self.lookup = {s: i for i,s in enumerate(self.chars)}
        self.rlookup = {i: s for i,s in enumerate(self.chars)}
        self.bigram = self.process_bigram()
        self.prob_bigram = (self.bigram+1).float() # Probability distribution of bigram | Smoothed out to avoid prob of 0
        self.prob_bigram /= self.prob_bigram.sum(1, keepdim = True)  # Broadcasting!
    def __repr__(self):
        sample = self.generate()
        return f"HeuristicBigram({sample=})"
    def __call__(self, char: str = ''):
        return self.generate(char)
    
    def _process_bigram(self, n: int = 0) -> dict:
        """Returns a dictionary version of the bigram."""
        b = dict()
        data = self.data
        for w in data[:n if n != 0 else len(data)]:
            chars = ['/'] + list(w) + ['/']
            for c1,c2 in zip(chars,chars[1:]):
                bigram = (c1,c2)
                b[bigram] = b.get(bigram, 0) + 1
        return b
    
    def _view_common(self, n: int = 1, d: dict = None, ):
        """Returns the n most common pairs of a dictionary. Defaults to the bigram's data."""
        if not d: d = self._process_bigram()
        x = sorted(d.items(),key=lambda x: -x[1])[:n]
        for k,v in x:
            print(f"{k[0]+k[1]}: {v}")

    def process_bigram(self) -> torch.tensor:
        """Process the bigram and returns a pytorch tensor."""
        data = self.data
        N = torch.zeros((27,27), dtype=torch.int32) # Counts of bigrams (26 letters + 2 special chars)
        for w in data:
            chars = ['/'] + list(w) + ['/']
            for c1,c2 in zip(chars,chars[1:]):
                i1, i2 = self.lookup[c1],self.lookup[c2]
                N[i1,i2] += 1
        return N
    
    def heatmap(self, basic: bool = False):
        """Plots a heatmap on the data's bigrams."""
        array = self.bigram
        if basic:
            plt.imshow(array)
        else:
            plt.figure(figsize=(16,16))
            plt.imshow(array, cmap = 'Blues')
            for i in range(27):
                for j in range(27):
                    char = self.rlookup[i] + self.rlookup[j]
                    plt.text(j,i,char,ha="center",va="bottom",color="gray")
                    plt.text(j,i,array[i,j].item(),ha="center",va="top",color="gray")
            plt.axis('off')

    def cost(self, data: list = []) -> float:
        """Calculates the average neg log likelihood of the model."""
        log_likelihood = 0.0
        n = 0
        if not data: data = self.data
        for w in data:
            chars = ['/'] + list(w) + ['/']
            for c1, c2 in zip(chars, chars[1:]):
                i1, i2  = self.lookup[c1], self.lookup[c2]
                prob = self.prob_bigram[i1, i2] # Probability of predicting the second given the first
                logprob = torch.log(prob)
                log_likelihood += logprob
                n += 1
        nll = -log_likelihood
        return nll/n # Normalized
        # Maximum Likelihood Estimation: Probability of reproducing entire dataset given model
        #! GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
        # Equivalent to maximizing the log likelihood (log is monotonic)
        # Equivalent to minimizing the negative log likelihood
        # Equivalent to minimizing the average log likelihood
        #* log (a*b*c) = log(a) + log(b) + log(c)

    def generate(self, char: str = '') -> str:
        """Generates and returns a bigram constructed from the data."""
        # Simply chooses next character based on most likely (as probability)!
        output = char
        index = self.lookup[char[-1]] if char else 0 # Begin the word (start token)
        while True:
            p = self.prob_bigram[index] # Get corresponding row
            index = torch.multinomial(p, num_samples=1, replacement=True).item() # Sample 1

            if index == 0: # End token
                break
            output += self.rlookup[index]

        return output
    
    def generateN(self, n: int = 10) -> list:
        result = []
        for _ in range(n):
            result.append(self.generate())
        return result

    
