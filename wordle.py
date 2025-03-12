import random
from rich.console import Console

CHARS = """abcdefghijklmnopqrstuvwxyz"""

class Wordle:
    def __init__(self, file: str = 'wordle.txt', minlen: int = 5, maxlen: int = 5, lives: int = 6, valid: bool = True, stats: bool = True):
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
        self.mostly = 'bold gold1'
        self.wrong = 'bold grey78'
        self.avg = [0,0,0] if stats else []
    def __repr__(self):
        st = '' if not self.avg else f', stats=[{self.avg[0]},{self.avg[1]},{self.avg[2]/self.avg[0] if self.avg[0] else 0}]'
        return f"Wordle(words={len(self.database)}{st})"
    def play(self, word: str = ""):
        """Play the game itself. Input a word to use it and override length limitations."""
        self.tries = 0
        word = word if word else random.choice(self.database) if self.database else 'world'
        self.console.print(f"[turquoise2]Welcome to Wordle![/turquoise2] You know the rules already, word has length {len(word)}.")
        correct = False
        cancel = False
        while self.tries < self.lives:
            show = ["_" for _ in range(len(word))]
            self.console.print(" ".join(show))
            try:
                guess = input('Guess: ')
            except: return
            if not guess:
                cancel = True
                break
            guess = guess.lower()
            if not self.check_guess(word, guess):
                print("Invalid guess!")
                continue
            right = 0
            for i,c in enumerate(guess):
                if c in word and word[i] == c:
                    right += 1
                    color = self.correct
                elif c in word:
                    color = self.mostly
                else:
                    color = self.wrong
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
            print(f'Cancelled, the word was {word}.')
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
    
    def check_guess(self, word: str, guess: str) -> bool:
        """Check whether guess is valid."""
        v = all(c in CHARS for c in guess)
        if self.valid: v = v and guess in self.database
        return  len(guess) == len(word) and v
    
    def update(self, file: str = "") -> list:
        """Update database with file."""
        data = set()
        if getattr(self, 'database', False): data.update(self.database)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                for word in line.split():
                    if self.minlen <= len((word:=word.lower())) <= self.maxlen:
                        if all(c in CHARS for c in word): data.add(word)
        return list(data)

class Hangman:
    def __init__(self, file: str = 'wenglish.txt', lives: int = 6, stats: bool = True):
        """Console-based hangman"""
        self.lives = lives
        self.tries = 0
        self.console = Console()
        self.avg = [0,0,0] if stats else []
        self.database = self.update(file)
    
    def play(self, word: str = ''):
        word = word if word else random.choice(self.database) if self.database else 'jazz'
        self.tries = 0
        attempts = 0
        self.console.print(f"[turquoise2]Welcome to Hangman![/turquoise2]\nGuess the word:")
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
                self.console.print(f'[red1]{guess.upper()}[/red1] is wrong!')
                self.tries += 1
            self.console.print(' '.join(show))
            self.console.print(f'Lives: {self.tries}/{self.lives} | Attempts: {attempts}')
        if cancel:
            print(f'Cancelled, the word was {word}.')
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
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()
                if all(c in CHARS for c in word): data.add(word)
        return list(data)
