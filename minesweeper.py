
import pygame,random,os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Difficulty:
# Beginner - 9x9, 10 mines
# Intermediate - 16x16, 40 mines
# Expert - 16x30, 99 mines

class Cell:
    def __init__(self, index: tuple, mine: bool = False, shown: bool = False, win = None):
        self.shown = shown
        self.hint = -1 if mine else 0
        self.index = index
        self.win = win
    def __repr__(self):
        return str(self.hint) if self.hint != -1 else 'X'
    def check(self, board: list) -> None:
        """Checks adjacent cells and edits self.hint"""
        self.hint = 0
        for dx,dy in [(0,1),(0,-1),(1,-1),(1,0),(1,1),(-1,-1),(-1,0),(-1,1)]:
            x,y = self.index[0]+dx,self.index[1]+dy
            if 0<=x<len(board[0]) and 0<=y<len(board) and board[y][x].hint == -1:
                self.hint += 1

class Board:
    def __init__(self, w: int, h: int, mines: int = 0, win = None):
        self.w = w
        self.h = h
        self.mines = mines
        self.win = win
        self.board = [[Cell((x,y),win=win) for x in range(self.w)] for y in range(self.h)]
    def setup(self, firstclick: tuple | None = None, mines: int | None = None) -> None:
        """Sets up the mines and hints. """
        if firstclick is None: firstclick = (0,0)
        if mines is None: mines = self.mines
        mines = min(mines,self.w*self.h-1)
        choices = [self.board[h][w] for w in range(self.w) for h in range(self.h) if (w,h)!=firstclick]
        random.shuffle(choices)
        for _ in range(mines):
            choices.pop().hint = -1
        for c in choices+[self.board[firstclick[1]][firstclick[0]]]:
            c.check(self.board)
