import pygame
from random import randint

LOAD = False # Load initial state from a file (insert file name in CUSTOM)
CUSTOM = "maze2.txt"
if LOAD: LOAD = CUSTOM

N = 25
if LOAD:
    with open(LOAD,'r') as f:
        N = len(f.readline())-1
ALIVE_STATE = "#"
DEAD_STATE = "."

WIDTH, HEIGHT = 800, 800
WIDTH, HEIGHT = WIDTH + (N - WIDTH % N), HEIGHT + (N - HEIGHT % N)
w, h = WIDTH//N, HEIGHT//N
POSX = POSY = 0
win = None
CELLCOLOR = "red"
CELL_BORDER = "#a10505"
BG = "white"
BG_BORDER = "#827979"
BORDERSIZE = 1

DEFAULT_SPEED = 3
OVERPOPULATION = 4
UNDERPOPULATION = 1
REPRODUCTION = 3

N_THEMES = 6
def theme(n: int = 0):
    """0: Default | 1: Dark Mode | 2: Aqua | 3: Neon | 4: Retro | 5: Random"""
    global CELLCOLOR,CELL_BORDER,BG,BG_BORDER
    match n:
        case 0:
            CELLCOLOR = "red"
            CELL_BORDER = "#a10505"
            BG = "white"
            BG_BORDER = "#827979"
        case 1:
            CELLCOLOR = "#e0e0e0"
            CELL_BORDER = "#b0b0b0"
            BG = "#0d0d0d"
            BG_BORDER = "#42446b"
        case 2:
            CELLCOLOR = "#0077be"
            CELL_BORDER = "#005f9e"
            BG = "#e0f7fa"
            BG_BORDER = "#b2ebf2"
        case 3:
            CELLCOLOR = "#39ff14"
            CELL_BORDER = "#ff073a"
            BG = "#000000"
            BG_BORDER = "#8a2be2"
        case 4:
            CELLCOLOR = "#00ff00"
            CELL_BORDER = "#004d00"
            BG = "#000000"
            BG_BORDER = "#1a1a1a"
        case 5:
            CELLCOLOR = (randint(0,255),randint(0,255),randint(0,255))
            CELL_BORDER = (randint(0,255),randint(0,255),randint(0,255))
            BG = (randint(0,255),randint(0,255),randint(0,255))
            BG_BORDER = (randint(0,255),randint(0,255),randint(0,255))

class Cell:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y
        self.alive = False
        self.neighbors = 0
        self.directions = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)]
        self.pos = (POSX + self.x * w, POSY + self.y * h, w, h)

    def draw(self, color: str = None, border: int = 0):
        if not color:
            if self.alive:
                pygame.draw.rect(win, CELLCOLOR, self.pos)
                if BORDERSIZE: pygame.draw.rect(win, CELL_BORDER, self.pos, BORDERSIZE)
            else:
                pygame.draw.rect(win, BG, self.pos)
                if BORDERSIZE: pygame.draw.rect(win, BG_BORDER, self.pos, BORDERSIZE)
        else:
            pygame.draw.rect(win, color, self.pos, border)

    def get_neighbors(self, board):
        n = 0
        for dx,dy in self.directions:
            x, y = self.x+dx, self.y+dy
            if 0<=x<N and 0<=y<N and board.grid[x][y].alive: n += 1
        self.neighbors = n
    
    def update_neighbors(self, board):
        for dx,dy in self.directions:
            x, y = self.x+dx, self.y+dy
            if 0<=x<N and 0<=y<N:
                c = board.grid[x][y]
                if self.alive: board.grid[x][y].neighbors += 1
                elif c.neighbors > 0: board.grid[x][y].neighbors -= 1

class Grid:
    def __init__(self):
        self.grid = self.makegrid()
        self.cells = 0
        self.ticks = 0
        self.start = [[0 for i in range(N)] for j in range(N)]

    def makegrid(self) -> list:
        grid = [0 for _ in range(N)]
        for i in range(N):
            grid[i] = [Cell(i,j) for j in range(N)]
        return grid
    
    def draw(self):
        win.fill(BG)
        for i in range(N):
            for j in range(N):
                self.grid[i][j].draw()
        pygame.display.update()
    
    def savestate(self):
        self.start = []
        for i in range(N):
            row = []
            for j in range(N):
                row.append(ALIVE_STATE) if self.grid[i][j].alive else row.append(DEAD_STATE)
            self.start.append(row)
    
    def override(self):
        for i in range(N):
            for j in range(N):
                self.grid[i][j].alive = True if self.start[i][j] == ALIVE_STATE else False
        for i in range(N):
            for j in range(N):
                self.grid[i][j].get_neighbors(self)
    
    def update(self):
        self.cells = 0
        for i in range(N):
            for j in range(N):
                self.grid[i][j].get_neighbors(self)
                if self.grid[i][j].alive: self.cells += 1

    def touch(self, pos: tuple):
        x, y = pos
        cell = self.grid[x // w][y // h]
        if not cell.alive:
            cell.alive = True
            cell.draw()
            cell.update_neighbors(self)
            self.cells += 1
            pygame.display.update()

    def kill(self, pos: tuple):
        x, y = pos
        cell = self.grid[x // w][y // h]
        if cell.alive:
            cell.alive = False
            cell.draw()
            cell.update_neighbors(self)
            self.cells -= 1
            pygame.display.update()

    def info(self, pos: tuple):
        x, y = pos
        cell = self.grid[x // w][y // h]
        print(f"Cell {cell.x},{cell.y} | {'alive' if cell.alive else 'dead'} | {cell.neighbors} neighbors")

    def generation(self):
        live = []
        dead = []
        for i in range(N):
            for j in range(N):
                c = self.grid[i][j]
                if c.neighbors == REPRODUCTION and not c.alive: 
                    live.append(c)
                elif (c.neighbors >= OVERPOPULATION or c.neighbors <= UNDERPOPULATION) and c.alive:
                    dead.append(c)
        for c in live:
            c.alive = True
            c.draw()
            c.update_neighbors(self)
            self.cells += 1
        for c in dead:
            c.alive = False
            c.draw()
            c.update_neighbors(self)
            self.cells -= 1
        self.ticks += 1

    def printstate(self):
        for row in self.start:
            print("".join(map(str,[x for x in row])))

    def loadfile(self, filename):
        self.start = []
        with open(filename,'r') as f:
            for line in f:
                self.start.append(list(map(lambda x: int(x) if x.isdigit() else x, line)))

class Game():
    def __init__(self):
        global win
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Game of Life")
        self.grid = Grid()
        self.drawing = True
        self.run = False
        self.clock = pygame.time.Clock()
        self.speed = DEFAULT_SPEED
        self.start = []
        self.paused = False
        self.theme = 0
        if LOAD:
            self.grid.loadfile(LOAD)
            self.grid.override()

    def mainloop(self):
        global BORDERSIZE
        self.grid.draw()
        self.grid.update()
        while self.drawing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    try:
                        self.grid.touch(pos)
                    except (AttributeError,IndexError):
                        continue
                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    try:
                        self.grid.kill(pos)
                    except (AttributeError,IndexError):
                        continue
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return False
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_q:
                        self.drawing = False
                        self.run = True
                        self.grid.savestate()
                        self.paused = False
                        break
                    elif event.key == pygame.K_i:
                        pos = pygame.mouse.get_pos()
                        try:
                            self.grid.info(pos)
                        except (AttributeError,IndexError):
                            continue
                    elif event.key == pygame.K_c:
                        for i in range(N):
                            for j in range(N):
                                self.grid.grid[i][j].alive = False
                        self.grid.update()
                        self.grid.draw()
                        pygame.display.update()
                    elif event.key == pygame.K_p:
                        print(f"Population: {self.grid.cells} | Generations: {self.grid.ticks} | Game speed: {self.speed}")
                    elif event.key == pygame.K_s:
                        self.grid.savestate()
                        self.grid.printstate()
                    elif event.key == pygame.K_l:
                        self.speed += 1
                    elif event.key == pygame.K_k and self.speed > 1:
                        self.speed -= 1
                    elif event.key == pygame.K_b:
                        BORDERSIZE = 0 if BORDERSIZE else 1
                        self.grid.draw()
                    elif event.key == pygame.K_h:
                        print("""Space: Start/stop | X: Return to initial state | I: Cell info
                    P: General info | L/K: +/- speed | S: Save grid state | B: Toggle border
                    C: Clear board | D: Change theme | Q: Quit""")
                    elif event.key == pygame.K_d:
                        self.theme = (self.theme + 1) % N_THEMES
                        theme(self.theme)
                        self.grid.draw()

        while self.run:
            self.clock.tick(self.speed)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return False
                    elif event.key == pygame.K_SPACE:
                        self.run = False
                        self.drawing = True
                        self.grid.ticks = 0
                        self.paused = True
                        break
                    elif event.key == pygame.K_x:
                        self.run = False
                        self.drawing = True
                        self.grid.ticks = 0
                        self.grid.override()
                        self.paused = True
                    elif event.key == pygame.K_i:
                        pos = pygame.mouse.get_pos()
                        try:
                            self.grid.info(pos)
                        except (AttributeError,IndexError):
                            continue
                    elif event.key == pygame.K_p:
                        print(f"Population: {self.grid.cells} | Generations: {self.grid.ticks} | Game speed: {self.speed}")
                    elif event.key == pygame.K_ESCAPE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_l:
                        self.speed += 1
                    elif event.key == pygame.K_k and self.speed > 1:
                        self.speed -= 1
                    elif event.key == pygame.K_s:
                        self.grid.savestate()
                        self.grid.printstate()
                    elif event.key == pygame.K_b:
                        BORDERSIZE = 0 if BORDERSIZE else 1
                        self.grid.draw()
                    elif event.key == pygame.K_h:
                        print("""Space: Start/stop | X: Return to initial state | I: Cell info
                        P: General info | L/K: +/- speed | S: Save grid state | B: Toggle border | D: Change theme""")
                    elif event.key == pygame.K_d:
                        self.theme = (self.theme + 1) % N_THEMES
                        theme(self.theme)
                        self.grid.draw()
                        
            if not self.paused: self.grid.generation()
            pygame.display.update()
        return True

    def loop(self):
        while self.mainloop():
            pass

if __name__ == "__main__":
    print("Welcome to Conway's Game of Life. Press H to view the keybinds.")
    g = Game()
    g.loop()
