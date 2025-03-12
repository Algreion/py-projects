import random
from rich.console import Console
import os

CHARS = """abcdefghijklmnopqrstuvwxyz"""
WORDLEFILE = 'wordle.txt'
HANGMANFILE = 'wenglish.txt'

class Wordle:
    def __init__(self, file: str = os.path.realpath(WORDLEFILE), minlen: int = 5, maxlen: int = 5, lives: int = 6, valid: bool = True, stats: bool = True):
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
        st = '' if not self.avg else f', stats=[{self.avg[0]},{self.avg[1]},{self.avg[2]/self.avg[0] if self.avg[0] else 0:.3f}]'
        return f"Wordle(words={len(self.database)}{st})"
    def play(self, word: str = ""):
        """Play the game itself. Input a word to use it and override length limitations."""
        self.tries = 0
        flag = bool(word)
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
            if not self.check_guess(word, guess, flag):
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
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    for word in line.strip().split():
                        if all(c in CHARS for c in word): data.add(word)
        finally: return list(data)

def gameloop():
    wordle = None
    hangman = None
    run = True
    playing = 0
    cancel = False
    while run:
        if not playing:
            Console().print('[dodger_blue2]Which word game do you want to play?[/dodger_blue2]')
            print('1. Wordle','2. Hangman',sep='\n')
            inp = input('> ')
            if inp == '1':
                if not wordle: wordle = Wordle()
                playing = wordle
            elif inp == '2':
                if not hangman: hangman = Hangman()
                playing = hangman
            else:
                cancel = True
                run = False
                break
        gm = "Wordle" if playing == wordle else "Hangman"
        print(f'Currently playing {gm} | Options:')
        print(f'1. Play {gm}',f'2. Customize {gm}',f'3. View {gm} stats.', '4. Custom Game', '5. Return', '6. Quit', sep='\n')
        inp = input('> ')
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
                print('\n'*20)
                playing.play(word)
            case '2':
                print('1. Lives','2. MinLen', '3. MaxLen', '4. Valid Check', '5. Update Data',sep=' | ')
                inp = input('> ')
                match inp:
                    case '5':
                        file = input('Input a file name in the same folder: ')
                        try:
                            playing.database = playing.update(file)
                        except:
                            print('File not found')
                            continue
                    case '4':
                        choice = input('1 to turn on, 0 to turn off. If off, any string of characters is accepted.')
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
                    case _:
                        choice = input('Choose a new number of lives: ')
                        try: playing.lives = int(choice)
                        except:
                            print('Invalid, defaulting to 6.')
                            playing.lives = 6
            case _:
                cancel = True
                run = False
                break

    if cancel:
        print('Quitting... Thanks for playing!')
        return
    
if __name__ == '__main__':
    gameloop()
