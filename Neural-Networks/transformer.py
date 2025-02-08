import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Data:
    def __init__(self, file: str = '', context: int = 10):
        """Character-level transformer architecture."""
        try:
            with open(file,'r',encoding='utf-8') as f:
                self.data = f.read()
        except:
            self.data = ''
        self.lookup = dict(enumerate(['']+sorted(list(set(self.data))))) # index -> char
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
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        return inputs,labels # tuple


class BigramLM(Data, nn.Module):
    def __init__(self, file: str = ''):
        """Basic Bigram character model, simplified with pytorch."""
        Data.__init__(self,file)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)
        with torch.no_grad():
            self.embedding.weight *= 0.1
        self.optimizer = torch.optim.AdamW(self.embedding.parameters(), lr=1e-3)

    def __call__(self, word: str = '', number: int = 0, maxlen: int = 100):
        n = number
        if type(word)==int: word, number = '', word
        if type(n)==str: word = n
        return self.generateN(number, word, maxlen) if number > 0 else self.generate(word, maxlen)
    
    def forward(self, inputs, labels = None) -> tuple:
        """Returns the logits and loss of the model."""
        logits = self.embedding(inputs) # [Batch,Context,VocabSize] | Prediction of char given previous
        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C) # Every row is different example's logits.
            labels = labels.view(-1) # Flatten labels
            loss = F.cross_entropy(logits, labels)
        return logits,loss
    
    def train(self, n: int = 1, info: bool = False, batchsize: int = 32):
        for i in range(n):
            X, Y = self.trainingset(batchsize)
            _, loss = self.forward(X,Y)
            if info and (not i%1000 or i==n-1): print(f'{i}. {loss.item():.3f}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def generate(self, word: str = '', maxlen: int = 100) -> str:
        """Generate one word based on training data."""
        if not word: context = torch.zeros((1,),dtype=torch.long,device=DEVICE)
        else: context = torch.tensor(self.encode(word),device=DEVICE)
        for i in range(maxlen):
            logits, _ = self.forward(context[-1]) # Feed last character
            probs = F.softmax(logits, dim = 0)
            index = torch.multinomial(probs, 1)
            if index == 0: break
            context = torch.cat((context,index), 0) # Add one to context (till maxlen)
        return self.decode(context.tolist())

    def generateN(self, n: int = 10, word: str = '', maxlen: int = 100) -> list:
        """Generate a list of N words based on training data."""
        res = []
        for _ in range(n):
            res.append(self.generate(word, maxlen))
        return res

if __name__=='__main__':
    b = BigramLM('shakespeare.txt')
    b = b.to(DEVICE)
