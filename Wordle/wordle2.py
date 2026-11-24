
import random,os

# TODO: Add help (at program start), way to change mode (det/rand), pool size, count attempts

DIR = os.path.dirname(os.path.abspath(__file__))

class Wordle:
    def __init__(self, file: str = DIR+'\\wordle.txt', valid: str = DIR+'\\wordle-valid.txt'):
        self.file = file
        self.file_valid = valid
        self.words = []
        self.words_extended = []
        self.addwords()
        self.frequency = self.frequent()
    
    def addwords(self, replace: bool = True) -> None:
        if replace or not isinstance(self.words,list): self.words = []
        try:
            with open(self.file,'r',encoding='utf-8') as f:
                for word in f.readlines():
                    self.words.append(word.lower().strip())
        except OSError as e:
            print(f"Unable to read file, error: {e}")
        if self.file_valid:
            try:
                with open(self.file_valid,'r',encoding='utf-8') as f:
                    for word in f.readlines():
                        self.words_extended.append(word.lower().strip())
            except OSError: self.words_extended = self.words

    def valid(self, tries: dict | None = None, extended: bool = False) -> list:
        """
        Given tries as 'words: 01120' where 0=gray,1=yellow,2=green;
        returns list of valid words.
        """
        if tries is None: tries = {}
        res = []
        if extended: words = self.words_extended
        else: words = self.words
        for word in words:
            val = True
            for attempt,result in tries.items():
                if self.turn(word,attempt) != result:
                    val = False
                    break
            if val: res.append(word)
        if not res and not extended: return self.valid(tries=tries,extended=True)
        return res
    
    def guess(self, tries: dict | None = None, mode: int = 1, pool: int = 10):
        """Gives a guess based on previous tries. Modes:
        1. Naive approach
        2. Naive approach - Random
        3. ???
        4. ??? - Random
        """
        if tries is None: tries = {}
        g = 'world'
        if mode in [1,2]:
            options = self.bestguess(tries,length=pool)
            if options: g = random.choice(options) if mode==2 else options[0]
            else: g = 'N/A'
        else:
            if not tries or (len(tries)<3 and any(att.count('0')>2 for att in tries.values())):
                options = self.bestguess(tries,extended=True,length=pool)
                if options: g = random.choice(options) if mode==2 else options[0]
                else: g = 'N/A'
            else:
                valid = self.valid(tries)
                if 2<=len(valid)<=10:
                    stuff = ''.join(set(''.join(w for w in valid))-set(''.join(w for w in tries)))
                    options = self.bestguess(tries,mode=2,mode_param=stuff,length=pool)
                    return random.choice(options) if mode==4 else options[0]
                options = self.bestguess(tries,length=pool)
                if options: g = random.choice(options) if mode==4 else options[0]
                else: g = 'N/A'
        return g

    def bestguess(self, tries: dict | None = None, mode: int = 1, mode_param: any = None, extended: bool = False, length: int | None = None) -> list:
        """
        Returns a list of the n=length best guesses. Modes:
        1. Finds a word with unique and common letters. Best for first and second guess.
        2. Finds a word with the most letters among those in mode_param. Best to weed out options.
        """
        if tries is None: tries = {}
        if tries: words = self.valid(tries,extended)
        else: words = self.words_extended if extended else self.words
        done = {}
        points = None
        if mode == 1 or (mode==2 and mode_param is None):
            ranks = sorted(self.frequency, key=self.frequency.get)
            points = lambda word: 0 if len(word)!=len(set(word)) else sum(ranks.index(c) for c in word)
        elif mode == 2:
            words = self.words
            points = lambda word: sum(c in mode_param for c in set(word))
        else:
            points = lambda word: 0
        for word in words:
            done[word] = points(word)
        return sorted(done, key=done.get, reverse=True)[:length]

    def turn(self, word: str, attempt: str) -> str:
        """Returns the green, yellow and gray results."""
        if len(word) != len(attempt): return ''
        res = ['0' for _ in range(len(word))]
        rem = {c:word.count(c) for c in ''.join(set(attempt))}
        for i,(w,a) in enumerate(zip(word,attempt)):
            if w==a:
                res[i] = '2'
                rem[w] -= 1
        for i,a in enumerate(attempt):
            if res[i]=='0' and a in rem and rem[a]>0:
                res[i] = '1'
                rem[a]-=1
        return ''.join(res)

    def frequent(self, tries: dict | None = None) -> dict:
        if tries is None: tries = {}
        f = dict()
        if tries: words = self.valid(tries)
        else: words = self.words
        for word in words:
            for c in word:
                f[c] = 1+f.get(c,0)
        return f
    
    def solve(self, mode: int = 1, pool: int = 10, tries: dict | None = None, attempts: list | None = None) -> bool:
        if tries is None: tries = {}
        if attempts is None: attempts = []
        attempts = attempts[::-1]
        res = '0'*5
        g = ''
        done = []
        while res != '2'*5:
            if attempts:
                g = attempts.pop()
                if len(g) != 5: g = self.guess(tries=tries,mode=mode,pool=pool)
            else: g = self.guess(tries=tries,mode=mode,pool=pool)
            print(f"\n{g}")
            try:
                res = input('> ').lower()
            except:
                return False
            if not res:
                return False
            if res in ['u','undo']:
                W = done.pop()
                del tries[W]
                print(f"Removed '{W}' from guesses.")
                continue
            elif res in ['won','w','win']:
                res = '22222'
            elif res in ['reset','r']:
                tries = {}
                print("Resetting...")
                continue
            elif res in ['options','opt','o']:
                l = self.bestguess(tries)
                print(f"Possible options: {len(l)}")
                print(f"Options: {', '.join(l[:10])}{f', (...and {len(l)-10} others)' if len(l)>10 else ''}")
                continue
            elif res in ['valid','val','v']:
                l = self.bestguess(tries,extended=True)
                print(f"Valid guesses: {len(l)}")
                print(f"Options: {', '.join(l[:10])}{f', (...and {len(l)-10} others)' if len(l)>10 else ''}")
                continue
                continue
            elif len(res) == 11 and res[5]==' ':
                adding1,adding2 = res.split()
                tries[adding1] = adding2
                done.append(adding1)
                print(f"Adding attempt '{adding1}' -> {adding2}.")
                continue
            elif len(res)!=5 or any(c not in '012' for c in res):
                print("\nInvalid output. Press Enter to quit, or try again (0=gray, 1=yellow, 2=green; eg. 00121)")
                res = '0'*5
                continue
            tries[g] = res
            done.append(g)
        print(f"\nWon after {len(tries)} attempts!")
        x = input("Again? (y/n)\n> ")
        return x.lower() in ['y','1']
    
    def automatic_solve(self, word: str, mode: int = 1, pool: int = 10, tries: dict | None = None) -> list:
        if tries is None: tries = {}
        attempts = []
        res = '0'*5
        while res != '2'*5:
            g = self.guess(tries=tries,mode=mode,pool=pool)
            res = self.turn(word,g)
            if not res: return []
            attempts.append(g)
            tries[g] = res
        return attempts
    
    def test(self, mode: int, n: int | None = 1000, info:bool = True) -> float:
        count = len(self.words) if n is None else n
        total = 0
        for i,word in enumerate(self.words[:n]):
            if i>0 and (not (i+1)%100 or i+1==count) and info: print(f'{i+1}/{count}: {word}')
            l = self.automatic_solve(word=word,mode=mode)
            if not l: print(f"Unable to solve '{word}'!")
            total += len(l)
        return total/count

if __name__ == '__main__':
    w = Wordle()
    run = True
    mode = 2 # 1 deterministic, 2 random
    pool = 25
    while run:
        run = w.solve(mode=mode,pool=pool)
    print("\nQuitting...")