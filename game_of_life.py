import pygame

N = 20
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

class Cell:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y
        self.alive = False
        self.neighbors = 0
        self.directions = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)]

    def draw(self, color: str = None, border: int = 0):
        if not color:
            if self.alive:
                pygame.draw.rect(win, CELLCOLOR, (POSX + self.x * w, POSY + self.y * h, w, h))
                pygame.draw.rect(win, CELL_BORDER, (POSX + self.x * w, POSY + self.y * h, w, h), BORDERSIZE)
            else:
                pygame.draw.rect(win, BG, (POSX + self.x * w, POSY + self.y * h, w, h))
                pygame.draw.rect(win, BG_BORDER, (POSX + self.x * w, POSY + self.y * h, w, h), BORDERSIZE)
        else:
            pygame.draw.rect(win, color, (POSX + self.x * w, POSY + self.y * h, w, h), border)

    def getneighbors(self, board):
        n = 0
        for dx,dy in self.directions:
            x, y = self.x+dx, self.y+dy
            if 0<=x<N and 0<=y<N and board.grid[x][y].alive: n += 1
        self.neighbors = n
    def giveneighbors(self, board):
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
                row.append(1) if self.grid[i][j].alive else row.append(0)
            self.start.append(row)
    
    def override(self):
        for i in range(N):
            for j in range(N):
                self.grid[i][j].alive = True if self.start[i][j] == 1 else False
        for i in range(N):
            for j in range(N):
                self.grid[i][j].getneighbors(self)
    def update(self):
        self.cells = 0
        for i in range(N):
            for j in range(N):
                self.grid[i][j].getneighbors(self)
                if self.grid[i][j].alive: self.cells += 1
    def touch(self, pos: tuple):
        x, y = pos
        cell = self.grid[x // w][y // h]
        if not cell.alive:
            cell.alive = True
            cell.draw()
            cell.giveneighbors(self)
            self.cells += 1
            pygame.display.update()
    def kill(self, pos: tuple):
        x, y = pos
        cell = self.grid[x // w][y // h]
        if cell.alive:
            cell.alive = False
            cell.draw()
            cell.giveneighbors(self)
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
                if c.neighbors == 3 and not c.alive: 
                    live.append(c)
                elif (c.neighbors > 3 or c.neighbors < 2) and c.alive:
                    dead.append(c)
        for c in live:
            c.alive = True
            c.draw()
            c.giveneighbors(self)
            self.cells += 1
        for c in dead:
            c.alive = False
            c.draw()
            c.giveneighbors(self)
            self.cells -= 1
        self.ticks += 1

class Game():
    def __init__(self):
        global win
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Game of Life")
        self.grid = Grid()
        self.drawing = True
        self.run = False
        self.clock = pygame.time.Clock()
        self.speed = 3
        self.start = []
        self.paused = False

    def restart(self):
        self.__init__()

    def mainloop(self):
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
                    if event.key == pygame.K_SPACE:
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
                        print(f"Population: {self.grid.cells}")
                    elif event.key == pygame.K_s:
                        # Starting condition
                        self.grid.savestate()
                        for row in self.grid.start:
                            print(row)

        while self.run:
            self.clock.tick(self.speed)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
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
            if not self.paused: self.grid.generation()
            pygame.display.update()
        return True

    def loop(self):
        while self.mainloop():
            pass

if __name__ == "__main__":
    g = Game()
    g.loop()
