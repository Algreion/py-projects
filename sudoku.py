import pygame
import time
import random
from copy import deepcopy
pygame.font.init()

N = 9
BOARDS = 20

def logg(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"{time.perf_counter() - start}s")
        return res
    return wrapper

class ArraySudoku:
    def __init__(self, test=False):
        if test: # From Cracking the Cryptic: https://www.youtube.com/watch?v=9m9t8ie9-EE
            board = [[5,0,0,2,0,0,0,4,0],[0,0,0,6,0,3,0,0,0],[0,3,0,0,0,9,0,0,7],
                     [0,0,3,0,0,7,0,0,0],[0,0,7,0,0,8,0,0,0],[6,0,0,0,0,0,0,2,0],
                     [0,8,0,0,0,0,0,0,3],[0,0,0,4,0,0,6,0,0],[0,0,0,1,0,0,5,0,0]]
        self._board = [[0]*N for _ in range(N)] if not test else board
    
    def print_board(self):
        for r in range(N):
            print(self._board[r])

    def valid(self):
        rows, cols, boxes = dict(), dict(), dict()
        for r in range(N):
            for c in range(N):
                v = self._board[r][c]
                if v == 0: continue
                i = (c//3) + (r//3) * 3
                if ((r in rows and v in rows[r]) or (c in cols and v in cols[c]) 
                or (i in boxes and v in boxes[i])): return False
                rows.setdefault(r, set()).add(v)
                cols.setdefault(c, set()).add(v)
                boxes.setdefault(i, set()).add(v)
        return True
    
    def solved(self):
        for r in range(N):
            for c in range(N):
                if self._board[r][c] == 0: return False
        return self.valid()

    def solve(self):
        rows, cols, boxes = {}, {}, {}
        for r in range(N):
            for c in range(N): # Early saving of current cells, O(n^2) time
                v = self._board[r][c]
                if v != 0:
                    i = (c // 3) + (r // 3) * 3
                    rows.setdefault(r, set()).add(v)
                    cols.setdefault(c, set()).add(v)
                    boxes.setdefault(i, set()).add(v)
        def dfs(r, c):
            if r >= N: return True # Processed all rows
            next_r, next_c = (r, c+1) if c < 8 else (r+1, 0)
            if self._board[r][c] != 0: return dfs(next_r, next_c) # Skip done cells
            i = (c // 3) + (r // 3) * 3
            for n in range(1, N+1): # Trying valid values
                if (n in rows.get(r, set()) or n in cols.get(c, set()) or n in boxes.get(i, set())): continue
                self._board[r][c] = n
                rows.setdefault(r, set()).add(n)
                cols.setdefault(c, set()).add(n)
                boxes.setdefault(i, set()).add(n)
                if dfs(next_r, next_c): return True # Found solution
                self._board[r][c] = 0 # Backtracking
                rows[r].remove(n)
                cols[c].remove(n)
                boxes[i].remove(n)
            return False # No solution, try another path
        return dfs(0, 0)
    
    def timedsolve(self):
        start = time.time()
        attempts = 0
        giveup = False
        rows, cols, boxes = {}, {}, {}
        for r in range(N):
            for c in range(N): # Early saving of current cells, O(n^2) time
                v = self._board[r][c]
                if v != 0:
                    i = (c // 3) + (r // 3) * 3
                    rows.setdefault(r, set()).add(v)
                    cols.setdefault(c, set()).add(v)
                    boxes.setdefault(i, set()).add(v)
        def dfs(r, c):
            nonlocal attempts, giveup
            if giveup: return False
            attempts += 1
            if attempts % 1000 == 0 and time.time()-start >= 1: 
                giveup = True
                return False
            if r >= N: return True # Processed all rows
            next_r, next_c = (r, c+1) if c < 8 else (r+1, 0)
            if self._board[r][c] != 0: return dfs(next_r, next_c) # Skip done cells
            i = (c // 3) + (r // 3) * 3
            for n in range(1, N+1): # Trying valid values
                if (n in rows.get(r, set()) or n in cols.get(c, set()) or n in boxes.get(i, set())): continue
                self._board[r][c] = n
                rows.setdefault(r, set()).add(n)
                cols.setdefault(c, set()).add(n)
                boxes.setdefault(i, set()).add(n)
                if dfs(next_r, next_c): return True # Found solution
                self._board[r][c] = 0 # Backtracking
                rows[r].remove(n)
                cols[c].remove(n)
                boxes[i].remove(n)
            return False # No solution, try another path
        return dfs(0, 0)
    
    def generate(self, n=17):
        if not (17<=n<=81): return False
        empty = set([n-1 for n in range(N**2)])
        while n > 0:
            cell = random.choice(list(empty))
            for num in range(1, 9):
                self._board[cell//9][cell%9] = num
                if self.valid(): break
                self._board[cell//9][cell%9] = 0
            if self._board[cell//9][cell%9] != 0:
                empty.remove(cell)
                n -= 1
        return self.valid()

    def wipe(self):
        for r in range(N):
            for c in range(N):
                self._board[r][c] = 0

WIDTH, HEIGHT = 910, 910
BOARD_W, BOARD_H = WIDTH-100, HEIGHT-100
w, h = BOARD_W//N, BOARD_H//N
POSX, POSY = 50, 5

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku")
FONT = pygame.font.SysFont("comicsans", 50)
MISTAKEFONT = pygame.font.SysFont("verdana", 30)
PENCIL = pygame.font.SysFont("verdana", 20)
THRESHOLD = 5 
HINTS = False

def theme(palette=0):
    """0: Default | 1: Night | 2: Pastel | 3: Ocean"""
    global NCOLOR, SEL_COLOR, HINT_COLOR, PM_COLOR, PM_COLOR, PM_COLOR_BG, SOLVER_COLOR, BG, MAIN_COLOR, MISTAKES_COLOR, SOLVING_COLOR
    match palette:
        case 0:
            NCOLOR = (0,0,255) # Color of placed numbers
            SEL_COLOR = (153, 255, 255) # Color of selected cell
            HINT_COLOR = (225,251,255) # Positional hints color
            PM_COLOR = (0, 0, 97) # Pencilmark numbers color
            PM_COLOR_BG = (255,230,230) # Pencilmark cell color
            SOLVER_COLOR = (153,255,204)
            BG = (255,255,255) # Background
            MAIN_COLOR = (0, 0, 0) # Color of pre-placed numbers and board borders
            MISTAKES_COLOR = (255, 0, 0) # Color of red X's
            SOLVING_COLOR = (0, 153, 0) # Color of "Solving" message
        case 1:
            NCOLOR = (100, 210, 255)
            SEL_COLOR = (80, 80, 120)
            HINT_COLOR = (40, 40, 70)
            PM_COLOR = (180, 180, 220)
            PM_COLOR_BG = (80, 60, 90)
            SOLVER_COLOR = (50, 150, 100)
            BG = (10, 10, 30)
            MAIN_COLOR = (220, 220, 250)
            MISTAKES_COLOR = (255, 102, 134)
            SOLVING_COLOR = (100, 200, 120)
        case 2:
            NCOLOR = (119, 158, 203)
            SEL_COLOR = (207, 207, 196)
            HINT_COLOR = (241, 225, 255)
            PM_COLOR = (175, 215, 237)
            PM_COLOR_BG = (255, 239, 203)
            SOLVER_COLOR = (173, 216, 230)
            BG = (255, 250, 240)
            MAIN_COLOR = (128, 128, 128)
            MISTAKES_COLOR = (255, 182, 193)
            SOLVING_COLOR = (152, 251, 152)
        case 3:
            NCOLOR = (0, 105, 148)
            SEL_COLOR = (102, 205, 170)
            HINT_COLOR = (135, 206, 235)
            PM_COLOR = (25, 25, 112)
            PM_COLOR_BG = (224, 255, 255)
            SOLVER_COLOR = (70, 130, 180)
            BG = (240, 248, 255)
            MAIN_COLOR = (0, 0, 128)
            MISTAKES_COLOR = (178, 34, 34)
            SOLVING_COLOR = (32, 178, 170)
        case _: theme()

def test_boards():
    for number in range(1,2):
        board = [[] for _ in range(N)]
        match number:
                case 1: board= None
                case _: pass
        b = ArraySudoku()
        b._board = board
        start = time.time()
        if not b.solve(): print(number, "is unsolveable!")
        print(f"{number} took {time.time()-start}s.")

class Cell:
    def __init__(self, y, x):
        self.x, self.y = x, y
        self.selected = False
        self.val = 0
        self.solving = False
        self.solid = False
        self.pm = False
        self.pval = ""
        self.hint = False
    
    def draw(self, color):
        coord = (POSX + self.x * w, POSY + self.y * h)
        if self.selected:
            pygame.draw.rect(win, SEL_COLOR, (coord[0], coord[1], w, h))
            pygame.draw.rect(win, NCOLOR, (coord[0], coord[1], w, h), 2)
        elif self.hint:
            pygame.draw.rect(win, HINT_COLOR, (coord[0], coord[1], w, h))
            pygame.draw.rect(win, color, (coord[0], coord[1], w, h), 1)
        elif self.solving:
            pygame.draw.rect(win, SOLVER_COLOR, (coord[0], coord[1], w, h))
            pygame.draw.rect(win, color, (coord[0], coord[1], w, h), 2)
        elif self.pm:
            pygame.draw.rect(win, PM_COLOR_BG, (coord[0], coord[1], w, h))
            pygame.draw.rect(win, PM_COLOR, (coord[0], coord[1], w, h), 2)
        else:
            pygame.draw.rect(win, BG, (coord[0], coord[1], w, h)) # Cell
            pygame.draw.rect(win, color, (coord[0], coord[1], w, h), 1) # Cell border
        if self.val != 0:
            if self.pval: self.pval = ""
            if self.solid: v = FONT.render(str(self.val), True, color)
            else: v = FONT.render(str(self.val), True, NCOLOR)
            center = v.get_rect(center=(coord[0] + w//2, coord[1] + h//2))
            win.blit(v, center)
        if self.pval:
            r1, r2, r3 = "","",""
            for i in range(len(self.pval)):
                if 0<=i<3: r1 += self.pval[i]
                elif 3<=i<6: r2 += self.pval[i]
                else: r3 += self.pval[i]
            if r3: 
                r3 = PENCIL.render(r3, True, PM_COLOR)
                win.blit(r3,((coord[0]+10,coord[1]+50)))
            if r2:
                r2 = PENCIL.render(r2, True, PM_COLOR)
                win.blit(r2,((coord[0]+10,coord[1]+30)))
            if r1: 
                r1 = PENCIL.render(r1, True, PM_COLOR)
                win.blit(r1,((coord[0]+10,coord[1]+10)))
            
class Sudoku:
    def __init__(self, number=None):
        """Initializes the Sudoku board using Cell objects"""
        self.board = [[0]*N for _ in range(N)]
        self.number = number
        if number:
            match number:
                case 1: board= [[3,0,1,0,0,8,0,0,0],[4,0,2,3,7,0,1,8,9],[5,0,9,6,2,1,0,4,7], 
                                [0,2,8,0,0,0,0,7,3],[0,0,7,0,0,6,0,0,0],[0,4,0,2,0,0,5,0,1], # Easy 
                                [0,0,4,7,5,0,0,0,0],[0,1,0,0,0,0,0,5,0],[0,0,0,0,6,4,7,0,8]]
                case 2: board= [[0,7,2,0,0,0,0,8,9],[0,0,9,8,0,0,0,1,6],[8,3,1,0,9,4,7,0,0], 
                                [4,0,5,0,2,0,0,0,3],[9,0,0,0,5,7,1,0,5],[0,0,3,1,0,8,0,2,0], # Easy 
                                [0,0,0,0,0,9,2,3,0],[0,8,4,0,3,5,0,0,0],[0,0,0,0,8,1,5,4,0]]
                case 3: board= [[0,2,0,5,0,8,9,0,3],[6,8,0,1,9,0,0,0,0],[0,0,0,3,4,0,0,0,7],
                                [0,0,1,9,0,0,0,4,5],[0,0,0,0,0,0,8,0,0],[3,0,9,0,0,0,0,0,0], # Easy
                                [0,0,2,0,0,5,0,0,0],[0,0,0,7,0,0,1,6,0],[7,0,0,0,1,0,5,0,8]]
                case 4: board= [[0,9,0,0,0,2,0,1,0],[2,0,8,0,4,0,9,3,0],[7,0,3,1,0,6,8,0,0],
                                [0,0,0,3,0,0,1,4,5],[1,8,5,0,2,9,6,0,0],[0,7,4,0,0,1,2,0,8], # Easy
                                [0,0,0,2,0,0,0,8,0],[5,0,0,9,0,0,7,6,2],[8,0,0,6,0,3,0,0,0]]
                case 5: board= [[0,0,0,0,0,0,0,8,0],[0,4,0,6,2,0,5,7,9],[3,7,5,8,0,0,0,0,2],
                                [8,2,9,0,3,5,0,6,1],[4,0,0,9,0,7,8,0,5],[7,5,1,0,8,6,3,0,4], # Easy
                                [9,1,6,0,0,0,0,0,0],[0,0,7,1,0,0,0,3,0],[2,3,0,5,0,8,9,0,0]]
                case 6: board= [[0,9,0,0,0,0,2,0,0],[0,0,2,6,9,0,0,4,0],[0,0,0,4,8,2,0,0,1],
                                [0,8,1,0,0,7,0,0,4],[4,0,3,8,0,0,1,6,0],[0,0,0,5,1,0,0,7,0], # Medium
                                [0,2,8,0,4,0,0,0,0],[1,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,5,2]]
                case 7: board= [[7,0,0,0,0,8,0,0,3],[5,0,0,0,0,0,6,1,0],[0,0,0,6,5,3,0,0,7],
                                [0,0,8,0,1,0,3,0,0],[9,7,5,0,0,0,4,0,0],[0,0,2,9,0,0,0,0,0], # Medium
                                [0,0,0,0,7,0,0,0,0],[0,4,0,5,3,6,0,0,0],[0,0,0,0,0,0,2,0,0]]
                case 8: board= [[0,0,0,6,0,0,5,0,1],[6,5,1,2,4,7,0,0,0],[0,0,0,5,0,0,0,0,0],
                                [0,0,0,0,6,8,0,2,0],[2,7,0,0,0,0,4,8,0],[0,8,3,7,2,0,0,0,0], # Medium
                                [9,0,5,8,0,2,0,1,0],[0,3,0,0,0,0,0,5,0],[0,2,4,0,5,0,6,0,0]]
                case 9: board= [[3,8,0,0,5,0,0,4,0],[0,5,0,6,0,3,0,0,7],[0,0,0,0,0,4,0,3,1], 
                                [0,0,0,0,0,8,9,0,2],[6,0,3,0,0,0,0,0,0],[0,0,8,5,6,1,0,0,0], # Medium
                                [0,0,6,7,0,0,2,1,0],[0,0,7,0,3,2,4,0,0],[0,2,0,0,9,0,0,0,3]]
                case 10: board= [[0,4,9,0,0,0,0,3,0],[6,0,0,7,0,0,0,0,0],[0,0,0,0,0,0,7,0,4], 
                                [0,3,0,9,0,0,8,0,0],[0,7,6,0,0,1,0,4,0],[9,8,1,4,0,3,0,5,0], # Medium
                                [0,0,0,0,6,8,4,0,7],[5,2,0,0,9,0,0,8,0],[0,0,0,0,0,2,9,0,0]]
                case 11: board= [[0,8,6,0,9,0,0,0,1],[0,4,0,0,1,0,0,6,0],[0,0,9,0,0,7,0,0,0], 
                                [0,9,0,8,0,0,0,0,0],[3,5,0,2,0,6,0,7,9],[0,0,0,0,0,1,0,3,0], # Medium
                                [0,0,0,1,0,0,4,0,0],[0,6,0,0,5,0,0,1,0],[4,0,0,0,2,0,7,9,0]]
                case 12: board=[[0,0,2,0,0,0,0,0,0],[0,0,0,0,0,0,9,1,3],[0,9,0,3,0,0,0,5,0],
                                [0,0,0,1,8,0,0,4,0],[0,0,0,0,0,4,7,2,0],[0,7,3,0,0,0,0,0,0], # Hard
                                [7,0,0,0,0,0,0,0,0],[0,1,0,0,7,0,0,0,0],[6,8,0,0,0,0,4,0,9]]
                case 13: board=[[0,0,8,2,0,0,0,0,0],[0,9,0,0,1,0,0,0,4],[0,6,0,0,0,0,3,0,0],
                                [0,0,0,7,0,0,0,2,0],[5,0,0,0,0,0,0,0,9],[0,1,0,0,0,6,0,0,0], # Hard
                                [0,0,7,0,0,0,0,8,0],[2,0,0,0,4,0,0,1,0],[0,0,0,0,0,3,5,0,0]]
                case 14: board=[[8,0,0,6,0,0,0,0,0],[7,0,0,0,0,3,4,0,0],[0,0,4,0,7,0,0,0,0],
                                [0,0,0,5,9,0,0,1,7],[0,0,0,0,0,0,0,5,0],[6,0,0,0,1,0,0,9,0], # Hard
                                [0,0,0,0,0,0,0,6,9],[9,5,0,0,8,0,0,0,0],[0,0,0,0,0,2,3,0,0]]
                case 15: board=[[5,0,0,2,0,0,0,4,0],[0,0,0,6,0,3,0,0,0],[0,3,0,0,0,9,0,0,7],
                                [0,0,3,0,0,7,0,0,0],[0,0,7,0,0,8,0,0,0],[6,0,0,0,0,0,0,2,0], # Hard | From Cracking the Cryptic: https://www.youtube.com/watch?v=9m9t8ie9-EE
                                [0,8,0,0,0,0,0,0,3],[0,0,0,4,0,0,6,0,0],[0,0,0,1,0,0,5,0,0]]
                case 16: board=[[0,0,0,0,0,0,0,0,0],[0,2,0,9,0,0,3,8,0],[0,3,0,1,0,0,7,5,0], 
                                [0,4,8,0,2,0,0,0,0],[0,5,0,0,0,6,0,0,0],[7,6,0,5,0,0,4,1,0], # Hard | Cracking the Cryptic: https://www.youtube.com/watch?v=fjWOgJqRWZI
                                [4,0,0,0,0,3,0,0,0],[2,0,0,8,4,5,6,7,0],[0,7,5,2,0,0,0,0,0]]
                case 17: board=[[0,1,2,0,3,0,4,5,0],[5,6,0,0,0,0,0,0,0],[3,0,0,0,0,0,0,0,2], 
                                [0,7,0,0,1,5,0,0,0],[0,0,0,6,0,9,0,0,0],[0,0,0,4,2,0,0,8,0], # Hard | Cracking the Cryptic: https://www.youtube.com/watch?v=YoO12J51Irs
                                [1,0,0,0,0,0,0,0,3],[0,0,0,0,0,0,0,2,4],[0,8,3,0,4,0,5,7,0]]
                case 18: board=[[1,0,0,4,0,0,7,0,0],[0,2,0,0,5,0,0,8,0],[0,0,3,0,0,6,0,0,9],
                                [0,1,0,0,4,0,0,7,0],[0,0,2,0,0,5,0,0,8],[9,0,0,3,0,0,6,0,0], # Extreme | From Cracking the Cryptic: https://www.youtube.com/watch?v=8C-A7xmBLRU
                                [7,0,0,0,0,8,0,0,2],[8,0,0,2,0,0,9,0,0],[0,9,0,0,7,0,0,1,0]]
                case 19: board=[[3,0,0,1,0,0,0,0,0],[0,7,0,0,0,0,0,1,0],[0,0,0,0,0,5,0,0,4],
                                [5,4,0,3,0,0,0,0,0],[0,0,7,0,0,9,0,0,1],[0,6,0,2,5,0,0,0,0], # Extreme 
                                [0,0,0,0,1,6,0,2,0],[2,0,0,8,0,0,6,0,0],[0,0,0,0,0,7,0,5,9]]
                case 20: board=[[0,0,0,1,0,2,0,0,0],[0,6,0,0,0,0,0,7,0],[0,0,8,0,0,0,9,0,0],
                                [4,0,0,0,0,0,0,0,3],[0,5,0,0,0,7,0,0,0],[2,0,0,0,8,0,0,0,1], # Extreme | From Cracking the Cryptic: https://www.youtube.com/watch?v=Ui1hrp7rovw
                                [0,0,9,0,0,0,8,0,5],[0,7,0,0,0,0,0,6,0],[0,0,0,3,0,4,0,0,0]]
                case _: number = None
        for r in range(N):
            for c in range(N):
                self.board[r][c] = Cell(r, c)
                if number: 
                    self.board[r][c].val = board[r][c]      
                    if self.board[r][c].val != 0: self.board[r][c].solid = True  
        
    def draw(self, clock=None, won=False):
        win.fill((BG))
        for r in range(N):
            for c in range(N):
                self.board[r][c].draw(MAIN_COLOR)
        for i in range(N+1):
            border = 6 if i % 9 == 0 else 4 if i % 3 == 0 else 1 # Thicker border around boxes
            pygame.draw.line(win, MAIN_COLOR, (POSX, POSY + i * h), (POSX + BOARD_W, POSY + i * h), border) # Horizontal
            pygame.draw.line(win, MAIN_COLOR, (POSX + i * w, POSY), (POSX + i * w, POSY + BOARD_H), border) # Vertical
        if mistakes:
            if mistakes > THRESHOLD: x = MISTAKEFONT.render("Game over! Solving...", True, (0, 190, 200))
            else: 
                x = MISTAKEFONT.render("X"*mistakes, True, MISTAKES_COLOR)
            win.blit(x, (50, HEIGHT-60))
        num = MISTAKEFONT.render(str(self.number),True,(MAIN_COLOR)) if self.number else MISTAKEFONT.render("R",True,(MAIN_COLOR))
        win.blit(num, (5,5))
        if won:
            m, s = clock//60, clock%60
            clock = FONT.render(f"SOLVED! [{m:02d}:{s:02d}]", True, SOLVING_COLOR)
            win.blit(clock, (WIDTH-50-clock.get_width(), HEIGHT-80))
        elif clock == None: 
            clock = FONT.render("SOLVING", True, SOLVING_COLOR)
            win.blit(clock, (WIDTH-50-clock.get_width(), HEIGHT-80))
        else:
            clock = FONT.render(f"{(clock//60):02d}:{(clock%60):02d}", True, MAIN_COLOR)
            win.blit(clock, (WIDTH-50-clock.get_width(), HEIGHT-80))
        pygame.display.update()

    def valid(self):
        rows, cols, boxes = dict(), dict(), dict()
        for r in range(N):
            for c in range(N):
                v = self.board[r][c].val
                if v == 0: continue
                i = (c//3) + (r//3) * 3
                if ((r in rows and v in rows[r]) or (c in cols and v in cols[c]) 
                or (i in boxes and v in boxes[i])): return False
                rows.setdefault(r, set()).add(v)
                cols.setdefault(c, set()).add(v)
                boxes.setdefault(i, set()).add(v)
        return True

    def solve(self):
        attempts = 0
        rows, cols, boxes = {}, {}, {}
        for r in range(N):
            for c in range(N): # Early saving of current cells, O(n^2) time
                if self.board[r][c].val != 0 and not self.board[r][c].solid: self.board[r][c].val = 0
                v = self.board[r][c].val
                if v != 0:
                    i = (c // 3) + (r // 3) * 3
                    rows.setdefault(r, set()).add(v)
                    cols.setdefault(c, set()).add(v)
                    boxes.setdefault(i, set()).add(v)
        def dfs(r, c):
            nonlocal attempts
            attempts += 1
            if attempts <= 100:
                self.draw()
                pygame.event.pump()
                time.sleep(0.01)
            elif attempts <= 1000 and not attempts % 5:
                self.draw()
                pygame.event.pump()
                time.sleep(0.01)
            elif attempts <= 10**4 and not attempts % 100:
                self.draw()
                pygame.event.pump()
            elif not attempts % 500:
                self.draw()
                pygame.event.pump()
            if r >= N: return True # Processed all rows
            next_r, next_c = (r, c+1) if c < 8 else (r+1, 0)
            if self.board[r][c].val != 0: return dfs(next_r, next_c) # Skip done cells
            i = (c // 3) + (r // 3) * 3
            for n in range(1, N+1): # Trying valid values
                if (n in rows.get(r, set()) or n in cols.get(c, set()) or n in boxes.get(i, set())): continue
                self.board[r][c].val = n
                rows.setdefault(r, set()).add(n)
                cols.setdefault(c, set()).add(n)
                boxes.setdefault(i, set()).add(n)
                self.board[r][c].solving = True
                if dfs(next_r, next_c): return True # Found solution
                self.board[r][c].solving = False
                self.board[r][c].val = 0 # Backtracking
                rows[r].remove(n)
                cols[c].remove(n)
                boxes[i].remove(n)
            return False # No solution, try another path
        return dfs(0, 0)
    
    def generate(self, n=17):
        if not (17<=n<=81): return False
        empty = set([n-1 for n in range(N**2)])
        while n > 0:
            cell = random.choice(list(empty))
            nums = [_+1 for _ in range(9)]
            random.shuffle(nums)
            for num in nums:
                self.board[cell//9][cell%9].val = num
                if self.valid(): break
                self.board[cell//9][cell%9].val = 0
            if self.board[cell//9][cell%9].val != 0:
                empty.remove(cell)
                n -= 1
                self.board[cell//9][cell%9].solid = True
        return self.valid()
    
    def generate_test(self, n=17, attempts = 10):
        good = False
        for _ in range(attempts):
            board = ArraySudoku()
            board.generate(n)
            save = deepcopy(board._board)
            if board.timedsolve():
                good = True
                break
            if n > 20: n -= 1
        if not good:
            print("Couldn't generate, defaulting!")
            return False
        for r in range(N):
            for c in range(N):
                self.board[r][c].val = save[r][c]
                if save[r][c] != 0: self.board[r][c].solid = True
        return True

    def clear(self):
        for r in range(N):
            for c in range(N):
                if self.board[r][c].val != 0 and not self.board[r][c].solid: 
                    self.board[r][c].val = 0
                    self.board[r][c].solving = False
    def wipe(self):
        for r in range(N):
            for c in range(N):
                self.board[r][c] = Cell(r, c)

    def select(self, pos):
        x, y = pos[0], pos[1]
        j, i = (x-POSX)//w, (y-POSY)//h
        cell = self.board[i][j]
        for r in range(N):
            for c in range(N):
                if self.board[r][c].selected or self.board[r][c].pm or self.board[r][c].hint: self.board[r][c].selected, self.board[r][c].pm, self.board[r][c].hint = False, False, False
        cell.selected = True
        if HINTS:
            row, col, box = i, j, (i//3) + (j//3) * 3
            for r in range(N):
                for c in range(N):
                    if r == row or c == col or (r//3) + (c//3) * 3 == box: self.board[r][c].hint = True

    def pencilmark(self, pos):
        x, y = pos[0], pos[1]
        j, i = (x-POSX)//w, (y-POSY)//h
        cell = self.board[i][j]
        for r in range(N):
            for c in range(N):
                if self.board[r][c].selected or self.board[r][c].hint: self.board[r][c].selected, self.board[r][c].hint = False, False
        if not cell.solid: cell.pm = True

mistakes = 0
FAIL: True
number_keys = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
               pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9}

def main():
    global mistakes, HINTS, FAIL, RANDOM
    if not RANDOM: game = Sudoku(number=SUDOKU)
    else:
        game = Sudoku()
        if not game.generate_test(CLUES): game = Sudoku(number=random.randint(1,BOARDS))
    start = time.time()
    confirm = False
    solved = False
    palette = 1
    run = True
    while run:
        if not solved: clock = time.time() - start
        if mistakes > 5: 
            game.solve()
            print("Game Over!")
            mistakes = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not solved: # Solve
                    if not game.solve(): print("Unsolveable!")
                    else: 
                        print("Press R for a new board.")
                        solved = True
                elif event.key == pygame.K_RETURN: # Submit
                    won = True
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].val == 0:
                                won = False
                    if not won or not game.valid(): 
                        print("The board isn't solved!")
                        continue
                    for r in range(N):
                        for c in range(N):
                            if not game.board[r][c].solid or game.board[r][c].solving:
                                game.board[r][c].solid, game.board[r][c].solving = True, False
                    print("Correct! Press R to play again")
                    solved = True
                elif event.key == pygame.K_t: # Change theme
                    theme(palette)
                    palette += 1
                    if palette > 3: palette = 0
                elif event.key == pygame.K_c: # Clear board
                    game.clear()
                elif event.key == pygame.K_h: # Toggle hints
                    HINTS = not HINTS
                    print(f"Toggled position hints {"on" if HINTS else "off"}")
                elif event.key == pygame.K_g: # Toggle random generation
                    RANDOM = not RANDOM
                    print(f"Toggled random-generation {"on" if RANDOM else "off"}")
                elif event.key == pygame.K_r: # Randomize board
                    if confirm:
                        confirm = False
                        solved = False
                        start = time.time()
                        print("Generating a new board!")
                        game.wipe()
                        pygame.display.update()
                        if RANDOM:
                            game.number = None
                            if not game.generate_test(CLUES): game = Sudoku(number=random.randint(1,BOARDS))
                        else: game = Sudoku(number=random.randint(1,BOARDS))
                    else:
                        print("Are you sure? Press R again to generate a new board")
                        confirm = True
                elif event.key == pygame.K_f: # Toggle fail condition
                    FAIL = False if FAIL else True
                    print("Toggled fail condition on") if FAIL else print("Toggled fail condition off")
                    mistakes = 0
                elif event.key in number_keys:
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].selected and not game.board[r][c].solid:
                                game.board[r][c].val = number_keys[event.key]
                                if FAIL and not game.valid(): 
                                    game.board[r][c].val = 0
                                    mistakes += 1
                            elif game.board[r][c].pm:
                                if event.key == pygame.K_0:
                                    game.board[r][c].pval = ""
                                    continue
                                if str(number_keys[event.key]) not in game.board[r][c].pval: 
                                    x = game.board[r][c].pval + str(number_keys[event.key])
                                    x = sorted(x)
                                    game.board[r][c].pval = "".join(x)
                                else: 
                                    x = list(game.board[r][c].pval)
                                    x.remove(str(number_keys[event.key]))
                                    game.board[r][c].pval = "".join(x)

            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                try:
                    game.select(pos)
                except (AttributeError,IndexError):
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].selected or game.board[r][c].hint: game.board[r][c].selected, game.board[r][c].hint = False, False
                    continue
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                try: game.pencilmark(pos)
                except IndexError:
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].pm: game.board[r][c].pm = False
        game.draw(round(clock), solved)
    pygame.quit()


RANDOM = False # Generate a random Sudoku, quality may vary
CLUES = random.randint(25,40) # 25-40
SUDOKU = random.randint(1,BOARDS-3) # Premade high-quality Sudokus 1-17 (of increasing difficulty)

FAIL = False  # Show mistakes, 5+ will end the game. If False, press ENTER with a complete board to confirm placements.
THRESHOLD = 5 # How many mistakes are allowed
HINTS = True # Position hints

if __name__ == "__main__":
    theme()
    print("Keys\nRight Click: Pencilmark | T: Change Theme | R: Generate new board | ENTER: Submit\nH: Toggle Hints | C: Clear board | F: Toggle mistakes | G: Toggle random-gen\nSPACE: Solver")
    main()
