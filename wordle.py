import pygame, random
from rich.console import Console
pygame.font.init()


WIDTH = 600
HEIGHT = 800
CHARS = """abcdefghijklmnopqrstuvwxyz"""

BG = (255,255,255)
CORRECT = ()

class BasicWordle:
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
        print(f"Welcome to Wordle! You know the rules already, word has length {len(word)}.")
        correct = False
        while self.tries < self.lives:
            show = ["_" for _ in range(len(word))]
            self.console.print(" ".join(show))
            try:
                guess = input('Guess: ')
            except: return
            if not guess: break
            guess = guess.lower()
            if not self.check_guess(word, guess):
                print("Invalid guess!\n")
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
        if self.tries == 0:
            print('Cancelled.')
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

def mainloop():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    win.fill(BG)
    pygame.display.update()
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
                break
    print("Thanks for playing!")

# mainloop()
