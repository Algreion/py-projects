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
        rows, cols, boxes = {}, {}, {}
        for r in range(N):
            for c in range(N):
                v = self._board[r][c]
                if v != 0:
                    i = (c // 3) + (r // 3) * 3
                    rows.setdefault(r, set()).add(v)
                    cols.setdefault(c, set()).add(v)
                    boxes.setdefault(i, set()).add(v)
        def dfs(r, c):
            if r >= N: return True # Processed all rows
            next_r, next_c = (r, c+1) if c < 8 else (r+1, 0)
            if self._board[r][c] != 0: return dfs(next_r, next_c)
            i = (c // 3) + (r // 3) * 3
            for n in range(1, N+1):
                if (n in rows.get(r, set()) or n in cols.get(c, set()) or n in boxes.get(i, set())): continue
                self._board[r][c] = n
                rows.setdefault(r, set()).add(n)
                cols.setdefault(c, set()).add(n)
                boxes.setdefault(i, set()).add(n)
                if dfs(next_r, next_c): return True
                self._board[r][c] = 0
                rows[r].remove(n)
                cols[c].remove(n)
                boxes[i].remove(n)
            return False
        return dfs(0, 0)
