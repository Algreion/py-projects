import torch
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAINING_STEPS = 10000
BATCH = 32
BLOCK = 8
LEARNING = 1e-3
EMBEDIMS = 32
MAXTOKENS = 100
HEAD_SIZE = 32
HEAD_NUMBER = 4
SINGLE_HEAD_SIZE = HEAD_SIZE // HEAD_NUMBER # Usually multiple heads are smaller

DEBUGGING = False

class Data:
    def __init__(self, file: str = '', context: int = BLOCK):
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
    def __repr__(self):
        return f"Data(vocabSize={self.vocab_size},trainingData={len(self.traindata)},devData={len(self.devdata)})"
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
        """Basic Bigram character model, now with pytorch."""
        Data.__init__(self,file)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)
        with torch.no_grad():
            self.embedding.weight *= 0.15
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING)
    def __repr__(self):
        return f'BigramLM(sample="{self(maxlen=10)}",vocabSize={self.vocab_size},data={len(self.data)})'

    def __call__(self, word: str = '', number: int = 0, maxlen: int = MAXTOKENS):
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
    
    def train(self, n: int = TRAINING_STEPS, info: bool = False, batchsize: int = BATCH):
        for i in range(n):
            X, Y = self.trainingset(batchsize)
            _, loss = self.forward(X,Y)
            if info and (not i%1000 or i==n-1): print(f'{i}. {loss.item():.3f}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate_loss(self, n: int = 1000):
        """Validate loss over the dev dataset."""
        X, Y = self.trainingset(n, 'dev')
        _, loss = self.forward(X,Y)
        return loss
    
    @torch.no_grad()
    def generate(self, word: str = '', maxlen: int = MAXTOKENS) -> str:
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

    def generateN(self, n: int = 10, word: str = '', maxlen: int = MAXTOKENS) -> list:
        """Generate a list of N words based on training data."""
        res = []
        for _ in range(n):
            res.append(self.generate(word, maxlen))
        return res

class Head(nn.Module):
    def __init__(self, head_size: int = HEAD_SIZE, context: int = BLOCK):
        """Single head of self-attention"""
        super().__init__()
        self.key = nn.Linear(EMBEDIMS, head_size, bias=False) # Every char has info about itself
        self.query = nn.Linear(EMBEDIMS, head_size, bias=False) # Every char wants to know specific things
        self.value = nn.Linear(EMBEDIMS, head_size, bias=False) # True embedding that is refined by key-queries
        self.register_buffer('tril',torch.tril(torch.ones(context, context)))
        self.training = True # Only mask future tokens if training

    def forward(self, inp):
        B,T,C = inp.shape if self.training else (1, *inp.shape) # Batch size | Tokens/Context | Character vectors
        if DEBUGGING: print("\ninput:",inp[0],B,T,C)
        k = self.key(inp) # BxTxH | H = HeadSize
        q = self.query(inp) # BxTxH
        if DEBUGGING: print("\nkey:",k[0],k.shape)
        if DEBUGGING: print("\nquery:",q[0],q.shape)
        # Attention scores (affinities) as dot products
        weights = q @ k.transpose(-2,-1) * C**-0.5 # BxTxH @ BxHxT = BxTxT | Layer by layer, TxH @ HxT -> TxT | Dot product between all k-q, how relevant each char is to another
        if DEBUGGING: print("\nWeights 1:",weights[0],weights.shape)
        if self.training: weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # Cannot communicate with future tokens
        weights = F.softmax(weights, dim = -1)
        if DEBUGGING: print("\nSoftmaxed Weights:",weights[0],weights.shape)
        # Weighted aggregation of values (Values are true embedded tokens)
        v = self.value(inp)
        if DEBUGGING: print("Value:",v[0],v.shape)
        return weights @ v # BxTxT @ BxTxH = BxTxH

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention acting in parallel."""
    def __init__(self, n_heads: int = HEAD_NUMBER, head_size: int = HEAD_SIZE):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.training = True
    def forward(self, inp):
        for h in self.heads:
            h.training = self.training
        return torch.cat([h(inp) for h in self.heads],dim=-1)

class GPT(Data, nn.Module):
    def __init__(self, file: str = '', context: int = BLOCK):
        """GPT Transformer architecture."""
        Data.__init__(self,file, context)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(self.vocab_size, EMBEDIMS) # Embedding lookup for each character
        self.position_embedding = nn.Embedding(self.context, EMBEDIMS) # Embedding lookup for positions (from 0 to context)
        self.sa_heads = MultiHeadAttention(HEAD_NUMBER, SINGLE_HEAD_SIZE) # Multi-head attention (key-query-values)
        self.lm_head = nn.Linear(SINGLE_HEAD_SIZE*HEAD_NUMBER, self.vocab_size) # ^ Correct possibly skewed dimensions
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING) # Gradient descent but better

    def __repr__(self):
        return f'GPT(sample="{self(maxlen=10)}",vocabSize={self.vocab_size},context={self.context},data={len(self.data)})'
    def __call__(self, word: str = '', number: int = 0, maxlen: int = MAXTOKENS):
        n = number
        if type(word)==int: word, number = '', word
        if type(n)==str: word = n
        return self.generateN(number, word, maxlen) if number > 0 else self.generate(word, maxlen)
        
    def inference(self, inputs):
        if DEBUGGING: print("\nStarting inference step")
        if DEBUGGING: print("input:",inputs)
        self.sa_heads.training = False
        T = inputs.numel()
        token_embeddings = self.embedding(inputs) #  token into N-dimensional space.
        if DEBUGGING: print("Token embed:",token_embeddings.shape)
        pos_embeddings = self.position_embedding(torch.arange(T,device=DEVICE)) # Embed also its position.
        if DEBUGGING: print("Position embed:",pos_embeddings.shape)
        activation = token_embeddings + pos_embeddings # Sum ^ and ^^ to get all info about individual token
        refined = self.sa_heads(activation) # Refine it with self-attention (context of other tokens)
        if DEBUGGING: print("Refined:",refined,"\n")
        return self.lm_head(refined)
    
    def forward(self, inputs, labels) -> tuple:
        """Returns the logits and loss of the model."""
        self.sa_heads.training = True
        if DEBUGGING: print("\nStarting forward step")
        if DEBUGGING: print("inputs:",inputs,inputs.shape)
        if DEBUGGING: print("\nlabels:",labels,labels.shape)
        B, T = inputs.shape
        if DEBUGGING: print("B, T:",B,T)
        token_embeddings = self.embedding(inputs) # [Batch,Context,VocabSize] | Prediction of char given previous
        if DEBUGGING: print("\nToken embedding:",token_embeddings.shape)
        pos_embeddings = self.position_embedding(torch.arange(T, device=DEVICE))
        if DEBUGGING: print("\nPosition embedding:",pos_embeddings.shape)
        activation = token_embeddings + pos_embeddings
        if DEBUGGING: print("\nActivation:",activation.shape)
        refined = self.sa_heads(activation)
        if DEBUGGING: print("\nRefined:",refined[0],refined.shape)
        logits = self.lm_head(refined)
        if DEBUGGING: print("\nLogits:",logits[0][0],logits.shape)
        B, T, C = logits.shape
        logits = logits.view(B*T,C) # Every row is different example's logits.
        if DEBUGGING: print("\nFlattened logits:",logits[0],logits.shape)
        labels = labels.view(-1) # Flatten labels
        if DEBUGGING: print("\nFlattened labels:",labels,labels.shape)
        loss = F.cross_entropy(logits, labels)
        if DEBUGGING: print("\nLoss:",loss)
        return logits, loss
    
    def train(self, n: int = TRAINING_STEPS, info: bool = False, batchsize: int = BATCH, plot: bool = False):
        if plot: trackloss = []
        for i in range(n):
            X, Y = self.trainingset(batchsize)
            _, loss = self.forward(X,Y)
            if info and (not i%1000 or i==n-1): print(f'{i}. {loss.item():.3f}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if plot: trackloss.append(loss.item())
        if plot:
            if len(trackloss) % 10: t = torch.tensor(trackloss)
            else: t = torch.tensor(trackloss).view(-1, n//10).mean(1)
            plt.plot(t)

    @torch.no_grad()
    def validate_loss(self, n: int = 1000):
        """Validate loss over the dev dataset."""
        X, Y = self.trainingset(n, 'dev')
        _, loss = self.forward(X,Y)
        return loss
    
    @torch.no_grad()
    def generate(self, word: str = '', maxlen: int = MAXTOKENS) -> str:
        """Generate one word based on training data."""
        if not word: context = torch.zeros((1,), dtype=torch.long, device=DEVICE)
        else: context = torch.tensor(self.encode(word),device=DEVICE)
        for i in range(maxlen):
            logits = self.inference(context[-self.context:]) # Feed last character
            probs = F.softmax(logits[-1], dim=-1)
            index = torch.multinomial(probs, 1)
            if index == 0: break
            context = torch.cat((context,index), 0) # Add one to context (till maxlen)
        return self.decode(context.tolist())
    
    def generateN(self, n: int = 10, word: str = '', maxlen: int = MAXTOKENS) -> list:
        """Generate a list of N words based on training data."""
        res = []
        for _ in range(n):
            res.append(self.generate(word, maxlen))
        return res

if __name__=='__main__':
    g = GPT('shakespeare.txt')
    g = g.to(DEVICE)
