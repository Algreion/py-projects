
class Tokenizer:
    def __init__(self, maxChars: int = 1000, charsFile: str = '', charsDict: dict = {}):
        self.chars = charsDict
        self.num = max(self.chars)+1 if self.chars else 256
        self.maxChars = maxChars
        self.minNum = self.num
        self.k = dict()
        if charsFile:
            self.loadchars(charsFile)
    
    def __call__(self, item: list) -> str:
        """Tokenize/Decode depending on item's type."""
        if isinstance(item,str):
            tokens = list(item.encode('utf-8'))
            return self.merge(tokens)
        else:
            tokens = self.unmerge(item)
            return self.decode(tokens)

    def findpairs(self, tokens: list) -> dict:
        counts = {}
        for pair in zip(tokens,tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
            if pair[0] == pair[1]: counts[pair] += 1
        return counts

    def merge(self, tokens: list) -> list:
        counts = sorted(((v,k) for k,v in self.findpairs(tokens).items()))
        n = self.num
        while counts and n <= self.maxChars and counts[-1][0] >= 2:
            _, (c1,c2) = counts.pop()
            if (c1,c2) not in self.k:
                self.chars[n] = (c1,c2)
                num = n
                n += 1
            else: num = self.k[(c1,c2)]
            new = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == c1 and tokens[i+1] == c2:
                    new.append(num)
                    i += 2
                else:
                    new.append(tokens[i])
                    i += 1
            tokens = new
            counts = sorted(((v,k) for k,v in self.findpairs(tokens).items()))
        self.k = dict([(v,k) for k,v in self.chars.items()])
        if n >= self.maxChars:
            check = True
            while check and counts and counts[-1][0] >= 2:
                check = False
                new = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and (tokens[i],tokens[i+1]) in self.k:
                        new.append(self.k[(tokens[i],tokens[i+1])])
                        i += 2
                        check = True
                    else:
                        new.append(tokens[i])
                        i += 1
                tokens = new
        self.num = n
        
        return tokens
    
    def unmerge(self, tokens: list) -> list:
        check = True
        while check:
            check = False
            new = []
            for c in tokens:
                if c >= self.minNum:
                    check = True
                    new.extend(self.chars[c])
                else:
                    new.append(c)
            tokens = new
        return tokens
    def decode(self, tokens: list) -> str:
        """Decode a list of tokens."""
        return bytes(tokens).decode('utf-8')

    def savechars(self, file: str) -> None:
        """Save current chars dictionary into file."""
        with open(file, 'w') as f:
            for k,(a,b) in self.chars.items():
                f.write(f"{','.join(map(str,[k,a,b]))}\n")

    def loadchars(self, file: str) -> dict:
        """Loads chars from file."""
        chars = {}
        with open(file, 'r') as f:
            for line in f.readlines():
                k,a,b = map(int,line.split(','))
                chars[k] = (a,b)
        self.num = max(chars) + 1 if chars else 256
        self.chars = chars
