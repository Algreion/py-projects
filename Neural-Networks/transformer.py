import torch

class Transformer:
    def __init__(self, file: str = '', context: int = 10):
        """Character-level transformer architecture."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read()
        except:
            self.data = ''
        self.lookup = dict(enumerate(sorted(list(set(self.data))))) # index -> char
        self.rlookup = dict((v,k) for k,v in self.lookup.items()) # char -> index
        self.vocab_size = len(self.lookup)
        self.encode = lambda s: [self.rlookup[c] for c in s]
        self.decode = lambda x: "".join(self.lookup[i] for i in x)
        self.context = context
        self.data = torch.tensor(self.encode(self.data), dtype = torch.long)
        self.splitdata()
    
    def splitdata(self, n: float = 0.9):
        """Splits the data into training and development data."""
        n = int(len(self.data)*n)
        self.traindata = self.data[:n]
        self.devdata = self.data[n:]
    
    def trainingset(self, batch_size: int = 4, split: str = 'train'):
        """Generates a batch of training/validation data."""
        data = self.traindata if split=='train' else self.devdata if split=='dev' else self.data
        index = torch.randint(len(data)-self.context, (batch_size,))
        inputs = torch.stack([data[i:i+self.context] for i in index])
        labels = torch.stack([data[i+1:i+self.context+1] for i in index])
        return inputs,labels # tuple
