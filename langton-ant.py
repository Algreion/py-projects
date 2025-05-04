import pygame
from random import randint

NX, NY = 50, 50
WIDTH, HEIGHT = 800, 800
W, H = WIDTH//NX, HEIGHT//NY
WIN = None
STATE0 = "black"
STATE1 = "white"
BORDER = "gray80"
ANTCOLOR = "red"
BORDERSIZE = 1
DIRECTIONS = [(0,1),(1,0),(0,-1),(-1,0)] # 0 = up, 1 = right, 2 = down, 3 = left
ANTSIZE = min(W,H)//2 - 1

STARTDIR = 0 # -1 for random start dir
STARTPOS = (NX//2,NY//2) # (-1,y) for random start x pos

STARTING = True # Don't touch

class Cell:
    def __init__(self, x: int , y: int):
        self.x, self.y = x, y
        self.state = 1
        self.neighbors = {}
        self.pos = (self.x*W, self.y*H, W, H)
    
    def draw(self, color: str = None) -> None:
        if color is None:
            if self.state == 1:
                pygame.draw.rect(WIN, STATE1, self.pos)
            elif self.state == 0:
                pygame.draw.rect(WIN, STATE0, self.pos)
        else:
            self.draw.rect(WIN, color, self.pos)
        if BORDERSIZE: pygame.draw.rect(WIN, BORDER, self.pos, BORDERSIZE)
        if not STARTING: pygame.display.update()
    
    def getneighbors(self, board) -> None:
        n = {}
        for i,(dx, dy) in enumerate(DIRECTIONS):
            x,y = self.x+dx,self.y+dy
            if 0 <= x < NX and 0 <= y < NY: n[i] = board.grid[x][y]
        self.neighbors = n

class Grid:
    def __init__(self, n: tuple = (NX, NY)):
        self.makegrid(n)
        self.ant = Ant()
        self.ticks = 0
    
    def makegrid(self, n: tuple) -> None:
        nw,nh=n
        grid = [0 for _ in range(nw)]
        for i in range(nw):
            grid[i] = [Cell(i,j) for j in range(nh)]
        self.grid = grid
        for i in range(nw):
            for j in range(nh):
                grid[i][j].getneighbors(self)
    
    def draw(self):
        for i in range(NX):
            for j in range(NY):
                self.grid[i][j].draw()
        self.ant.draw()

class Ant:
    def __init__(self):
        self.direction = STARTDIR if STARTDIR >= 0 else randint(0,len(DIRECTIONS)-1)
        self.x = STARTPOS[0] if STARTPOS[0] >= 0 else randint(0,NX-1)
        self.y = STARTPOS[1] if STARTPOS[1] >= 0 else randint(0,NY-1)

    def draw(self):
        pygame.draw.circle(WIN, ANTCOLOR, (STARTPOS[0]*W+W//2,STARTPOS[1]*H+H//2), ANTSIZE)
        if not STARTING: pygame.display.update()

def mainloop():
    global WIN, BORDERSIZE, STARTING
    run = True
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Langton's Ant")
    game = Grid()
    game.draw()
    pygame.display.update()
    STARTING = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
                return

mainloop()
