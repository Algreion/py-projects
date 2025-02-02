import torch
from matplotlib import pyplot as plt

#! Bigram Language Model
# Considers 2 characters at a time
# Given 1 char, predict the next in the sequence. (a_ -> ab)
# We include starting and ending special characters.

class Bigram:
    def __init__(self, file: str):
        """Neural Network to predict next character given 2."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read().splitlines()
        except:
            self.data = ''
        self.chars = list('abcdefghijklmnopqrstuvwxyz')+['<S>','<E>']
        self.lookup = {s: i for i,s in enumerate(self.chars)}
        self.reverselookup = {i: s for i,s in enumerate(self.chars)}
    
    def _process_bigram(self, n: int = 0) -> dict:
        """Returns a dictionary version of the bigram."""
        b = dict()
        data = self.data
        for w in data[:n if n != 0 else len(data)]:
            chars = ['<S>'] + list(w) + ['<E>']
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
        N = torch.zeros((28,28), dtype=torch.int32) # Counts of bigrams (26 letters + 2 special chars)
        for w in data:
            chars = ['<S>'] + list(w) + ['<E>']
            for c1,c2 in zip(chars,chars[1:]):
                i1, i2 = self.lookup[c1],self.lookup[c2]
                N[i1,i2] += 1
        return N
    
    def heatmap(self, basic: bool = False):
        array = self.process_bigram()
        if basic:
            plt.imshow(array)
        else:
            plt.figure(figsize=(16,16))
            plt.imshow(array, cmap = 'Blues')
            for i in range(28):
                for j in range(28):
                    char = self.reverselookup[i] + self.reverselookup[j]
                    plt.text(j,i,char,ha="center",va="bottom",color="gray")
                    plt.text(j,i,array[i,j].item(),ha="center",va="top",color="gray")
            plt.axis('off')

if __name__ == '__main__':
  b = Bigram("../words.txt")
