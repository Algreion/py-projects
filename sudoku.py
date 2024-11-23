import pygame

# Make board
# Solver
# Visualizer
# Game
# Generator

N = 9

class Sudoku:
    def __init__(self):
        self._board = [[0]*N for _ in range(N)]

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
        pass
