import pygame
import time
import random
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

class Sudoku:
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
