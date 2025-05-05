import pygame
from random import randint

# TODO | Add more states (colors), more ants, more directions (diagonals, still), etc.

NX, NY = 50, 50
REGSIZE = 800
if REGSIZE % NX != 0: REGSIZE+=(NX-REGSIZE%NX)
wh = (REGSIZE, REGSIZE)
if NX != NY: wh = (REGSIZE, round(REGSIZE/NX*NY)) if NX>NY else (round(REGSIZE/NY*NX),REGSIZE)
WIDTH, HEIGHT = wh
W, H = WIDTH//NX, HEIGHT//NY
WIN = None
STATE0 = "black"
STATE1 = "white"
BORDER = "gray80"
ANTCOLOR = "red"
REVERSANTCOLOR = "blue"
BORDERSIZE = 1
DIRECTIONS = [(0,-1),(1,0),(0,1),(-1,0)] # 0 = up, 1 = right, 2 = down, 3 = left
ANTSIZE = max(1, min(W,H)//2 - 1 )
STARTING = True # Don't touch
if NX>=100 or NY>=100: BORDERSIZE = 0

STARTSTATE = 1 # State for all starting cells
STARTDIR = 0 # -1 for random start dir
STARTPOS = (NX//2,NY//2) # (-1,y) for random start x pos
STARTRUNNING = False # If true, simulation starts immediately upon running the code.
INTROS = True # Start and end message while running

class Cell:
    def __init__(self, x: int , y: int):
        self.x, self.y = x, y
        self.state = STARTSTATE
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
        if BORDERSIZE > 0: pygame.draw.rect(WIN, BORDER, self.pos, BORDERSIZE)
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
    
    def reset(self, random: bool = False):
        global STARTING
        if random:
            for i in range(NX):
                for j in range(NY):
                    self.grid[i][j].state = STARTSTATE
                    self.ant.direction = randint(0,len(DIRECTIONS)-1)
                    self.ant.x, self.ant.y = randint(0,NX-1), randint(0,NY-1)
        else:
            for i in range(NX):
                for j in range(NY):
                    self.grid[i][j].state = STARTSTATE
                    self.ant.direction = STARTDIR
                    self.ant.x,self.ant.y = STARTPOS
        STARTING, self.ticks = True, 0
        self.draw()
        pygame.display.update()
        STARTING = False
    
    def tick(self):
        x,y = self.ant.x, self.ant.y
        cell = self.grid[x][y]
        if cell.state == 1:
            self.ant.direction = (self.ant.direction+1) % len(DIRECTIONS)
            cell.state = 0
            cell.draw()
        elif cell.state == 0:
            self.ant.direction = (self.ant.direction-1) % len(DIRECTIONS)
            cell.state = 1
            cell.draw()
        dx,dy = DIRECTIONS[self.ant.direction]
        self.ant.x, self.ant.y = (x+dx) % NX, (y+dy) % NY
        self.ant.draw()
        self.ticks += 1
    
    def reversetick(self):
        x,y = self.ant.x, self.ant.y
        dx,dy = DIRECTIONS[self.ant.direction]
        cell = self.grid[(x-dx)% NX][(y-dy) % NY]
        if cell.state == 1:
            self.ant.direction = (self.ant.direction+1) % len(DIRECTIONS)
            cell.state = 0
            cell.draw()
        elif cell.state == 0:
            self.ant.direction = (self.ant.direction-1) % len(DIRECTIONS)
            cell.state = 1
            cell.draw()
        self.grid[self.ant.x][self.ant.y].draw()
        self.ant.x, self.ant.y = (x-dx) % NX, (y-dy) % NY
        self.ant.draw()
        self.ticks -= 1

class Ant:
    def __init__(self):
        self.direction = STARTDIR if STARTDIR >= 0 else randint(0,len(DIRECTIONS)-1)
        self.x = STARTPOS[0] if STARTPOS[0] >= 0 else randint(0,NX-1)
        self.y = STARTPOS[1] if STARTPOS[1] >= 0 else randint(0,NY-1)

    def draw(self):
        pygame.draw.circle(WIN, ANTCOLOR, (self.x*W+W//2,self.y*H+H//2), ANTSIZE)
        if not STARTING: pygame.display.update()

def mainloop():
    global WIN, BORDERSIZE, STARTING, ANTCOLOR
    run = True
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Langton's Ant")
    game = Grid()
    game.draw()
    pygame.display.update()
    STARTING = False
    clock = pygame.time.Clock()
    paused = not STARTRUNNING
    spd = 4
    reverse = False
    a = ANTCOLOR
    SPDS = [1,2,3,5,10,20,30,45,60,0]
    if INTROS: print("Welcome to Langton's Ant! Press the spacebar to begin. (C for keybinds)")
    while run:
        clock.tick(SPDS[spd])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    run = False
                    break
                elif event.key == pygame.K_b:
                    BORDERSIZE = int(not BORDERSIZE)
                    STARTING = True
                    game.draw()
                    STARTING = False
                    pygame.display.update()
                elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_t:
                    game.tick()
                elif event.key == pygame.K_l:
                    spd = min(len(SPDS)-1,spd+1)
                elif event.key == pygame.K_k:
                    spd = max(0,spd-1)
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_q:
                    game.reset(True)
                elif event.key == pygame.K_y:
                    reverse = not reverse
                    if reverse: ANTCOLOR = REVERSANTCOLOR
                    else: ANTCOLOR = a
                elif event.key == pygame.K_f:
                    for i in range(NX):
                        for j in range(NY):
                            game.grid[i][j].state = not game.grid[i][j].state
                    STARTING = True
                    game.draw()
                    STARTING = False
                    pygame.display.update()
                elif event.key == pygame.K_i:
                    d = {}
                    for i in range(NX):
                        for j in range(NY):
                            ss = game.grid[i][j].state
                            d[ss] = d.get(ss, 0) + 1
                    print(f"Total ticks: {game.ticks} | White: {d.get(1,0)}, Black: {d.get(0,0)}")
                elif event.key == pygame.K_c:
                    print("""Keybinds:
    P/Space: Pause | K/L: Decrease/Increase speed | T: +1 step | I: Info
    R: Reset | Q: Random reset | C: Keybinds | F: Invert states | Y: Reverse time""")
        if not paused and pygame.get_init():
            if reverse and game.ticks > 0: game.reversetick()
            elif not reverse: game.tick()
    if INTROS: print(f"\nThanks for playing! [{game.ticks} ticks]")

mainloop()
