import random
from rich.console import Console
import os
from trie import Trie

CHARS = """abcdefghijklmnopqrstuvwxyz"""
WORDLEFILE = os.path.join(os.path.dirname(__file__),'wordle.txt')
HANGMANFILE = os.path.join(os.path.dirname(__file__),'wenglish.txt')

class Wordle:
    def __init__(self, file: str = WORDLEFILE, minlen: int = 5, maxlen: int = 5, lives: int = 7, valid: bool = True, stats: bool = True):
        """Console based Wordle, possible words taken from file.
        If valid set to false, the guess may be any letter combination."""
        self.console = Console()
        self.lives = lives
        self.tries = 0
        self.minlen = minlen
        self.maxlen = maxlen
        self.valid = valid
        self.database = self.update(file)
        self.correct = "bold green4"
        self.mostly = 'bold orange1'
        self.wrong = 'bold grey78'
        self.avg = [0,0,0] if stats else []
    def __repr__(self):
        st = '' if not self.avg else f', stats=[{self.avg[0]},{self.avg[1]},{self.avg[2]/self.avg[0] if self.avg[0] else 0:.3f}]'
        return f"Wordle(words={len(self.database)}{st})"
    def play(self, word: str = ""):
        """Play the game itself. Input a word to use it and override length limitations."""
        self.tries = 0
        flag = bool(word)
        word = word.lower() if word else random.choice(self.database) if self.database else 'world'
        self.console.print(f"\n[turquoise2]Welcome to Wordle![/turquoise2] You know the rules already, word has length {len(word)}.")
        correct = False
        cancel = False
        nums = dict([(v, 0) for v in CHARS])
        for c in word:
            nums[c] = nums.get(c,0)+1
        while self.tries < self.lives:
            found = {}
            show = ["_" for _ in range(len(word))]
            self.console.print(" ".join(show))
            try:
                guess = input('\nGuess: ')
            except: return
            if not guess:
                cancel = True
                break
            guess = guess.lower()
            if not self.check_guess(word, guess, flag):
                print("Invalid guess!")
                continue
            right = 0
            for i,c in enumerate(guess):
                if c in word and word[i] == c:
                    right += 1
                    found[c] = found.get(c, 0) + 1
                    col = f"[{self.correct}]" if self.correct else ''
                    col2 = f"[/{self.correct}]" if self.correct else ''
                    show[i] = f"{col}{c.upper()}{col2}"
            for i,c in enumerate(guess):
                if show[i] not in '_ ': continue
                if c in word and word[i] != c and (c not in found or found[c] < nums[c]):
                    color = self.mostly
                    found[c] = found.get(c,0)+1
                else: color = self.wrong
                col = f"[{color}]" if color else ''
                col2 = f"[/{color}]" if color else ''
                show[i] = f"{col}{c.upper()}{col2}"
            self.tries += 1
            self.console.print(''.join(show))
            if right == len(word):
                self.console.print(f"Congratulations, the word was [bold green1]{word.upper()}[/bold green1]! You took {self.tries} attempts!")
                correct = True
                break
            print(f'Attempts: {self.tries}/{self.lives}')
        if cancel:
            print(f'Cancelled, the word was {word}.\n')
            return
        if not correct:
            self.console.print(f"You lost! The word was [bold red1]{word.upper()}[/bold red1]")
        if self.avg:
            self.avg[0] += correct
            self.avg[1] += 1-correct
            self.avg[2] += self.tries if correct else 0

    def stats(self):
        """Show current stats, if enabled."""
        if not self.avg: return None
        avg = self.avg[2]/self.avg[0] if self.avg[0] else 0
        self.console.print(f"[chartreuse3]Correct: {self.avg[0]}[/chartreuse3]",f"[red3]Incorrect: {self.avg[1]}[/red3]",f"[deep_sky_blue1]Average attempts: {avg:.1f}[/deep_sky_blue1]",sep='\n')
    
    def check_guess(self, word: str, guess: str, flag: bool = False) -> bool:
        """Check whether guess is valid."""
        if word == guess: return True
        if flag: return len(guess) == len(word)
        v = all(c in CHARS for c in guess)
        if self.valid: v = v and guess in self.database
        return  len(guess) == len(word) and v
    
    def update(self, file: str = "") -> list:
        """Update database with file."""
        data = set()
        if getattr(self, 'database', False): data.update(self.database)
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    for word in line.strip().split():
                        if self.minlen <= len((word:=word.lower())) <= self.maxlen:
                            if all(c in CHARS for c in word): data.add(word)
        finally: return list(data)

class Hangman:
    def __init__(self, file: str = os.path.realpath(HANGMANFILE), lives: int = 6, stats: bool = True):
        """Console-based hangman"""
        self.lives = lives
        self.tries = 0
        self.console = Console()
        self.avg = [0,0,0] if stats else []
        self.database = self.update(file)
    def __repr__(self):
        st = '' if not self.avg else f', stats=[{self.avg[0]},{self.avg[1]},{self.avg[2]/self.avg[0] if self.avg[0] else 0:.3f}]'
        return f"Hangman(words={len(self.database)}{st})"
    
    def play(self, word: str = ''):
        word = word.lower() if word else random.choice(self.database) if self.database else 'jazz'
        self.tries = 0
        attempts = 0
        self.console.print(f"\n[turquoise2]Welcome to Hangman![/turquoise2]\nGuess the word:")
        correct = False
        show = ["_" if c != ' ' else ' ' for c in word]
        cancel = False
        while self.tries < self.lives:
            right = False
            self.console.print(" ".join(show))
            try:
                guess = input('Guess: ')
            except: return
            if not guess:
                cancel = True
                break
            attempts += 1
            guess = guess.lower()
            if len(guess) == 1:
                for i,c in enumerate(word):
                    if c == guess:
                        show[i] = f'[bold green1]{c.upper()}[/bold green1]'
                        right = True
            if (len(guess) != 1 and guess == word) or all(x != '_' or x == ' ' for x in show):
                for i in range(len(word)):
                    show[i] = f'[bold green1]{word[i].upper()}[/bold green1]'
                    right = True
                self.console.print(' '.join(show))
                self.console.print(f"Congratulations, the word was [bold green1]{word.upper()}[/bold green1]! You took {attempts} attempts!")
                correct = True
                break
            if not right:
                self.console.print(f'[red1]{guess.upper()}[/red1] is wrong!\n')
                self.tries += 1
            self.console.print(f'Lives: {self.tries}/{self.lives} | Attempts: {attempts}\n')
        if cancel:
            print(f'Cancelled, the word was {word}.\n')
            return
        if not correct:
            self.console.print(f"You lost! The word was [bold red1]{word.upper()}[/bold red1]")
        if self.avg:
            self.avg[0] += correct
            self.avg[1] += 1-correct
            self.avg[2] += attempts if correct else 0

    def stats(self):
        if not self.avg: return None
        avg = 0 if not self.avg[0] else self.avg[2]/self.avg[0]
        self.console.print(f"[green1]Correct: {self.avg[0]}[/green1]", f"[red1]Incorrect: {self.avg[1]}[/red1]", f"[deep_sky_blue1]Average attempts: {avg:.1f}[/deep_sky_blue1]",sep='\n')

    def update(self, file: str = "") -> list:
        """Update database with file."""
        data = set()
        if getattr(self, 'database', False): data.update(self.database)
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    for word in line.strip().split():
                        if all(c in CHARS for c in word): data.add(word)
        finally: return list(data)

class WordleSolver(Wordle):
    def __init__(self, file: str = WORDLEFILE, heuristic: int = 2, samples: int = 30, rush: int = 20, minlen: int = 5, maxlen: int = 5, lives: int = 7, valid: bool = True, stats: bool = True):
        """
        Heuristic: How many words it starts with before attempting to guess
        Samples: The amount of options it has at the start
        Rush: How many blitz games it will compute
        """
        self.trie = Trie()
        super().__init__(file,minlen,maxlen,lives,valid,stats)
        self.heuristic = heuristic
        self.samples = samples
        self.rush = rush
    
    def blitz(self, n: int = 0, word: str = ''):
        """Blitz through many games at once"""
        if not self.database:
            print("No words to choose from!")
            return
        if not n: n = self.rush
        WORD = None
        if word and word in self.database:
            WORD = word
        for atmpt in range(n):
            word = random.choice(self.database) if WORD is None else WORD
            GUESSES = []
            self.tries = 0
            nums = dict([(v, 0) for v in CHARS])
            for c in word: nums[c] = nums.get(c,0)+1
            HEUR = self.heuristic
            BANNED, CORRECT, POSITIONS, EXPLR = set(), {}, {}, ''
            correct = False
            while self.tries < self.lives:
                found = {}
                show = ["_" for _ in range(len(word))]
                if HEUR > 0:
                    OPTIONS = self.trie.uniqueword_common(len(word),''.join(EXPLR), self.samples)
                    guess = random.choice(OPTIONS) if OPTIONS else ''
                    HEUR -= 1
                    if len(CHARS)-len(EXPLR)-len(guess) > 0: EXPLR += guess
                    else: EXPLR = guess
                else:
                    guess = ['.' for _ in range(len(word))]
                    for i in CORRECT:
                        guess[i] = CORRECT[i]
                    OPTIONS = self.trie.wordle(''.join(guess), ''.join(set(i for x in POSITIONS for i in POSITIONS[x])), ''.join(BANNED))
                    OPTIONS = list(filter(lambda x: all(x[i] not in POSITIONS[i] for i in POSITIONS),OPTIONS))
                    guess = random.choice(OPTIONS)
                right = 0
                for i,c in enumerate(guess):
                    if c in word and word[i] == c:
                        right += 1
                        found[c] = found.get(c, 0) + 1
                        col = f"[{self.correct}]" if self.correct else ''
                        col2 = f"[/{self.correct}]" if self.correct else ''
                        show[i] = f"{col}{c.upper()}{col2}"
                        CORRECT[i] = c
                    elif c not in word:
                        BANNED.add(c)
                    else:
                        POSITIONS[i] = POSITIONS.get(i, []) + [c]
                for i,c in enumerate(guess):
                    if show[i] not in '_ ': continue
                    if c in word and word[i] != c and (c not in found or found[c] < nums[c]):
                        color = self.mostly
                        found[c] = found.get(c,0)+1
                    else: color = self.wrong
                    col = f"[{color}]" if color else ''
                    col2 = f"[/{color}]" if color else ''
                    show[i] = f"{col}{c.upper()}{col2}"
                self.tries += 1
                GUESSES.append(''.join(show))
                if right == len(word):
                    correct = True
                    break
            self.console.print(f'{atmpt+1}. {word.upper()}: {' '.join(g for g in GUESSES)} | {f'[bold green1]CORRECT[/bold green1] | {self.tries} attempts' if correct else '[bold red1]INCORRECT[/bold red1]'}')
            if self.avg:
                self.avg[0] += correct
                self.avg[1] += 1-correct
                self.avg[2] += self.tries if correct else 0

    def play(self, word: int = None):
        """Watch the computer play the game."""
        if not self.database:
            print("No words to choose from!")
            return
        if type(word) == str and word in self.database:
            word = self.database.index(word)
        elif type(word)==str and word.isdigit():
            word = int(word)
        if word is None or not (0<=word<len(self.database)):
            word = random.randint(0,len(self.database)-1)
        word = self.database[word]
        self.tries = 0
        self.console.print(f"\n[turquoise2]Welcome to Wordle![/turquoise2] You know the rules already, word has length {len(word)}.")
        correct = False
        nums = dict([(v, 0) for v in CHARS])
        for c in word: nums[c] = nums.get(c,0)+1
        HEUR = self.heuristic
        BANNED, CORRECT, POSITIONS, EXPLR = set(), {}, {}, ''
        while self.tries < self.lives:
            found = {}
            show = ["_" for _ in range(len(word))]
            self.console.print(" ".join(show))
            if HEUR > 0:
                OPTIONS = self.trie.uniqueword_common(len(word),''.join(EXPLR), self.samples)
                guess = random.choice(OPTIONS) if OPTIONS else ''
                HEUR -= 1
                if len(CHARS)-len(EXPLR)-len(guess) > 0: EXPLR += guess
                else: EXPLR = guess
            else:
                guess = ['.' for _ in range(len(word))]
                for i in CORRECT:
                    guess[i] = CORRECT[i]
                OPTIONS = self.trie.wordle(''.join(guess), ''.join(set(i for x in POSITIONS for i in POSITIONS[x])), ''.join(BANNED))
                OPTIONS = list(filter(lambda x: all(x[i] not in POSITIONS[i] for i in POSITIONS),OPTIONS))
                guess = random.choice(OPTIONS)
            if not guess:
                print("Something went wrong!")
                return
            print(f'> {guess}')
            right = 0
            for i,c in enumerate(guess):
                if c in word and word[i] == c:
                    right += 1
                    found[c] = found.get(c, 0) + 1
                    col = f"[{self.correct}]" if self.correct else ''
                    col2 = f"[/{self.correct}]" if self.correct else ''
                    show[i] = f"{col}{c.upper()}{col2}"
                    CORRECT[i] = c
                elif c not in word:
                    BANNED.add(c)
                else:
                    POSITIONS[i] = POSITIONS.get(i, []) + [c]
            for i,c in enumerate(guess):
                if show[i] not in '_ ': continue
                if c in word and word[i] != c and (c not in found or found[c] < nums[c]):
                    color = self.mostly
                    found[c] = found.get(c,0)+1
                else: color = self.wrong
                col = f"[{color}]" if color else ''
                col2 = f"[/{color}]" if color else ''
                show[i] = f"{col}{c.upper()}{col2}"
            self.tries += 1
            self.console.print(''.join(show))
            if right == len(word):
                self.console.print(f"Congratulations, the word was [bold green1]{word.upper()}[/bold green1]! You took {self.tries} attempts!")
                correct = True
                break
            print(f'Attempts: {self.tries}/{self.lives}')
        if not correct:
            self.console.print(f"You lost! The word was [bold red1]{word.upper()}[/bold red1]")
        if self.avg:
            self.avg[0] += correct
            self.avg[1] += 1-correct
            self.avg[2] += self.tries if correct else 0

        
    def update(self, file: str = "") -> list:
        """Update database and Trie with file."""
        data = set()
        if getattr(self, 'database', False): data.update(self.database)
        if not getattr(self, 'trie', False): self.trie = Trie()
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    for word in line.strip().split():
                        if self.minlen <= len((word:=word.lower())) <= self.maxlen:
                            if all(c in CHARS for c in word): data.add(word)
        finally: 
            self.trie.insertmulti(list(data))
            return list(data)

def gameloop():
    wordle, hangman, AI = None, None, None
    run = True
    playing = 0
    cancel = False
    while run:
        if not playing:
            Console().print('\n[dodger_blue2]Which word game do you want to play?[/dodger_blue2]')
            print('1. Wordle','2. Hangman', '3. Wordle AI',sep='\n')
            inp = input('> ')
            if inp == '1':
                if not wordle: wordle = Wordle()
                playing = wordle
            elif inp == '2':
                if not hangman: hangman = Hangman()
                playing = hangman
            elif inp == '3':
                if not AI: AI = WordleSolver()
                playing = AI
            else:
                cancel = True
                run = False
                break
        gm = "Wordle" if playing == wordle else 'Wordle AI' if playing == AI else "Hangman"
        Console().print(f'\nCurrently playing [light_steel_blue]{gm}[/light_steel_blue]\nOptions:')
        print(f'1. Play {gm}',f'2. Customize {gm}',f'3. View {gm} stats.', '4. Custom Game', '5. Return', '6. Quit', sep='\n')
        if AI == playing: print(f'7. AI Blitz ({playing.rush})')
        inp = input('\n> ')
        match inp:
            case '1':
                playing.play()
            case '5':
                playing = 0
                continue
            case '3':
                playing.stats()
            case '4':
                word = input('Choose a word: ')
                if playing == AI:
                    if word not in playing.database and not word.isdigit():
                        print('The word must be part of the database!')
                        word = None
                else: print('\n'*100)
                playing.play(word)
            case '2':
                print(f'\nCustomizing {gm}:','\n1. Lives','2. MinLen', '3. MaxLen', '4. Valid guess Checker', '5. Update Data','6. Go back', sep=' | ')
                if playing==AI: print('7. AI Heuristics | 8. AI Sample size | 9. AI Blitz')
                inp = input('> ')
                match inp:
                    case '5':
                        file = input('Input a file name in the same folder: ')
                        try:
                            w = len(playing.database)
                            playing.database = playing.update(file)
                            playing.console.print(f'Database updated successfully! Word count: {w} -> {len(playing.database)}\n')
                        except:
                            print('File not found')
                            continue
                    case '4':
                        choice = input('1 to turn on, 0 to turn off: ')
                        if choice == '1':
                            playing.valid = True
                        else: playing.valid = False
                    case '3':
                        choice = input('Input a new maximum length: ')
                        try: playing.maxlen = int(choice)
                        except:
                            print('Invalid, defaulting to 5.')
                            playing.maxlen = 5
                    case '2':
                        choice = input('Input a new minimum length: ')
                        try: playing.minlen = int(choice)
                        except:
                            print('Invalid, defaulting to 5.')
                            playing.minlen = 5
                    case '1':
                        choice = input('Choose a new number of lives: ')
                        try: playing.lives = int(choice)
                        except:
                            print('Invalid, defaulting to 6.')
                            playing.lives = 6
                    case '7':
                        if not playing==AI: continue
                        choice = input('Input a new heuristic (default=2): ')
                        try: playing.heuristic = int(choice)
                        except:
                            print('Invalid, defaulting to 2.')
                            playing.heuristic = 2
                    case '8':
                        if not playing==AI: continue
                        choice = input('Input a new sample size (default=30): ')
                        try: playing.samples = int(choice)
                        except:
                            print('Invalid, defaulting to 30.')
                            playing.samples = 30
                    case '9':
                        if not playing==AI: continue
                        choice = input('Input a new blitz amount: ')
                        try: playing.rush = int(choice)
                        except: 
                            print('Invalid, defaulting to 20.')
                            playing.rush = 20
                    case _:
                        continue
            case '7':
                playing.blitz()
                playing.stats()
            case _:
                cancel = True
                run = False
                break

    if cancel:
        print('\nQuitting... Thanks for playing!')
        return

if __name__ == '__main__':
    gameloop()
