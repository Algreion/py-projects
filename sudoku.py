import pygame
import time
import random
pygame.font.init()
# Make board DONE
# Solver DONE
# Generator DONE (kinda)
# Visualizer
# Game

N = 9

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

WIDTH, HEIGHT = 910, 910
BOARD_W, BOARD_H = WIDTH-100, HEIGHT-100
w, h = BOARD_W//N, BOARD_H//N
POSX, POSY = 50, 5

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku")
FONT = pygame.font.SysFont("comicsans", 50)
MISTAKEFONT = pygame.font.SysFont("verdana", 30)
THRESHOLD = 5

class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.selected = False
        self.val = 0
        self.solving = False
        self.solid = False
    
    def draw(self, color, bg="white"):
        coord = (POSX + self.x * w, POSY + self.y * h)
        if self.selected:
            pygame.draw.rect(win, (153, 255, 255), (coord[0], coord[1], w, h))
            pygame.draw.rect(win, "blue", (coord[0], coord[1], w, h), 2)
        elif self.solving:
            pygame.draw.rect(win, (153,255,204), (coord[0], coord[1], w, h))
            pygame.draw.rect(win, color, (coord[0], coord[1], w, h), 1)
        else:
            pygame.draw.rect(win, bg, (coord[0], coord[1], w, h)) # Cell
            if self.solid: pygame.draw.rect(win, "grey99", (coord[0], coord[1], w, h))
            pygame.draw.rect(win, color, (coord[0], coord[1], w, h), 1) # Cell border
        if self.val != 0: 
            v = FONT.render(str(self.val), True, color)
            center = v.get_rect(center=(coord[0] + w//2, coord[1] + h//2))
            win.blit(v, center)
            
class Sudoku:
    def __init__(self, test=False):
        """Initializes the Sudoku board using Cell objects"""
        self.board = [[0]*N for _ in range(N)]
        if test: # From Cracking the Cryptic: https://www.youtube.com/watch?v=9m9t8ie9-EE
            board = [[5,0,0,2,0,0,0,4,0],[0,0,0,6,0,3,0,0,0],[0,3,0,0,0,9,0,0,7],
                     [0,0,3,0,0,7,0,0,0],[0,0,7,0,0,8,0,0,0],[6,0,0,0,0,0,0,2,0],
                     [0,8,0,0,0,0,0,0,3],[0,0,0,4,0,0,6,0,0],[0,0,0,1,0,0,5,0,0]]
        for r in range(N):
            for c in range(N):
                self.board[r][c] = Cell(r, c)
                if test: 
                    self.board[r][c].val = board[r][c]      
                    if self.board[r][c].val != 0: self.board[r][c].solid = True  
        
    def draw(self):
        win.fill(("white"))
        for r in range(N):
            for c in range(N):
                self.board[r][c].draw("black")
        for i in range(N+1):
            border = 6 if i % 9 == 0 else 4 if i % 3 == 0 else 1 # Thicker border around boxes
            pygame.draw.line(win, "black", (POSX, POSY + i * h), (POSX + BOARD_W, POSY + i * h), border) # Horizontal
            pygame.draw.line(win, "black", (POSX + i * w, POSY), (POSX + i * w, POSY + BOARD_H), border) # Vertical
        if mistakes:
            if mistakes > THRESHOLD: x = MISTAKEFONT.render("Game over! Solving...", True, (0, 190, 200))
            else: 
                x = MISTAKEFONT.render("X"*mistakes, True, "red")
            win.blit(x, (50, HEIGHT-60))
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

    def solved(self):
        for r in range(N):
            for c in range(N):
                if self.board[r][c].val == 0: return False
        return self.valid()

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
            if attempts % 1000 == 0: 
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

    def select(self, pos):
        x, y = pos[0], pos[1]
        i, j = (x-POSX)//w, (y-POSY)//h
        cell = self.board[i][j]
        for r in range(N):
            for c in range(N):
                if self.board[r][c].selected: self.board[r][c].selected = False
        cell.selected = True

mistakes = 0
number_keys = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
               pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9}

def main():
    global mistakes
    game = Sudoku(test=True)
    run = True
    while run:
        if mistakes > 5: game.solve()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game.solve(): print("Unsolveable!")
                elif event.key == pygame.K_t: # Toggle states (cheat)
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].solid or game.board[r][c].solving:
                                game.board[r][c].solid, game.board[r][c].solving = False, False
                elif event.key == pygame.K_c: # Clear board
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].val != 0 and not game.board[r][c].solid: game.board[r][c].val = 0
                elif event.key in number_keys:
                    for r in range(N):
                        for c in range(N):
                            if game.board[r][c].selected and not game.board[r][c].solid:
                                game.board[r][c].val = number_keys[event.key]
                                if not game.valid(): 
                                    game.board[r][c].val = 0
                                    mistakes += 1

            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                try:
                    game.select(pos)
                except (AttributeError,IndexError):
                    continue
        game.draw()
    pygame.quit()

if __name__ == "__main__":
    main()
