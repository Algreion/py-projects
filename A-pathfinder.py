import pygame, heapq
from tkinter import messagebox
from time import perf_counter as perf
from math import sqrt
from random import choice

#? A* Pathfinding algorithm
# Unlike its predecessor Dijkstra, which only looks at the shortest path with a priority queue, this takes into
# consideration the distance from the end as well, with a heuristic estimate (Euclidean/Manhattan distance).
# The total cost thus is f = g + h, then it works the same as Dijkstra's algorithm with a priority queue

custom = False
TIMING = True
DISTANCE = 1 # 0 Euclidean, 1 Manhattan distance

def get_custom_grid(filename: str = "maze.txt") -> list:
    grid = []
    try:
        with open(filename, "r") as f:
            for line in f:
                grid.append(list(filter(lambda x: x in "SE#.", line)))
        return grid
    except: return []
if custom: CUSTOM_GRID = get_custom_grid()
else: CUSTOM_GRID = []

def mainloop():
    global CUSTOM_GRID, TIMING, DISTANCE
    SHOW = False
    CUSTOM = bool(CUSTOM_GRID)

    ST, ED = (0,0), (-1,-1)
    ROWS, COLS = 30, 30
    MODE = 0
    modes = ["normal", "king", "horse", "diagonals", "jumper", "flash", "wallhugger","drunk","mirror", "wormhole"]
    border = False

    class Node:
        def __init__(self, x, y):
            self.i = x
            self.j = y
            self.f = float("inf") # Cost function: f = g + h
            self.g = float("inf") # Cost from start node to current
            self.h = 0 # Heuristic estimate from current to end node
            self.neighbors = []
            self.previous = None
            self.obs = False # Obstacle
            self.processed = False
            self.value = 1 # Default cost value for g

        def show(self, color, border):
            if not self.processed:
                pygame.draw.rect(screen, color, (self.i * w, self.j * h, w, h), border)
                pygame.display.update()

        def path(self, color, border):
            pygame.draw.rect(screen, color, (self.i * w, self.j * h, w, h), border)
            pygame.display.update()

        def addNeighbors(self, grid):
            i = self.i
            j = self.j
            match modes[MODE]:
                case "king": dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
                case "horse": dirs = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1)]
                case "diagonals": dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
                case "jumper": dirs = [(1,0),(0,1),(-1,0),(0,-1),(2,0),(0,2),(-2,0),(0,-2)]
                case "flash":
                    dirss = [(1,0),(0,1),(-1,0),(0,-1)]
                    dirs = []
                    for di, dj in dirss:
                        ti,tj = i,j
                        while 0<=ti+di<COLS and 0<=tj+dj<ROWS and not grid[ti+di][tj+dj].obs:
                            ti+=di
                            tj+=dj
                        dirs.append((-i+ti,-j+tj))
                case "wallhugger":
                    dirs = []
                    for di,dj in [(1,0),(0,1),(-1,0),(0,-1)]:
                        for ddi, ddj in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                            if (di,dj) not in dirs and (not (0<=i+di+ddi<COLS) or not (0<=j+dj+ddj<ROWS) or grid[i+di+ddi][j+dj+ddj].obs): dirs.append((di,dj))
                case "drunk":
                    dirs = [(choice([-1, 0, 1]), choice([-1, 0, 1])) for _ in range(8)]
                case "mirror":
                    dirs = [(-i+j,-j+i),(0,-1),(0,1),(1,0),(-1,0)]
                case "wormhole":
                    dirs = [(1,0),(0,1),(-1,0),(0,-1)]
                    if i == 0: dirs.append((ROWS-1,j))
                    if j == 0: dirs.append((i,COLS-1))
                case _: dirs = [(1,0),(0,1),(-1,0),(0,-1)]
            
            for di, dj in dirs:
                if (0<=i+di<COLS) and (0<=j+dj<ROWS) and not grid[i+di][j+dj].obs:
                    self.neighbors.append(grid[i+di][j+dj])

    if CUSTOM:
        COLS, ROWS = len(CUSTOM_GRID), len(CUSTOM_GRID[0])
        grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                grid[r][c] = Node(r, c)
    else:
        grid = [[0 for _ in range(ROWS)] for _ in range(COLS)]
        # Create nodes
        for i in range(COLS):
            for j in range(ROWS):
                grid[i][j] = Node(i, j)

    BG = (0,7,20)
    CB = False # Cell Borders
    WALLS = (220,227,210)
    TOCHECK = (0,153,0) # (30,175,14)
    PROCESSED = (200,0,0)
    PATH = (0,110,245)
    START = (0,255,0)
    END = (255,0,255)
    BORDER = (160,160,160)
    EMPTY_S, OBS_S, PATH_S, START_S, END_S = ".","#","O","S","E"
    WIDTH = 800 + (COLS - 800%COLS)
    HEIGHT = 800 + (ROWS - 800%ROWS)
    w = h = min(WIDTH // COLS, HEIGHT // ROWS)

    if border:
        COLS += 2
        ROWS += 2
        if ST == (0, 0): ST = (1,1)
        if ED == (-1, -1): ED = (-2, -2)


    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A* Pathfinding")
    screen.fill(BG)

    toCheck = [] # Nodes to be evaluated
    processed = set() # Evaluated nodes

    # Show the empty grid
    if CB:
        for i in range(COLS):
            for j in range(ROWS):
                grid[i][j].show(WALLS, 1)

    if border:
        # Mark border (obstacles)
        for i in range(0, ROWS):
            grid[0][i].show(BORDER, 0) # Up
            grid[0][i].obs = True
            grid[-1][i].show(BORDER, 0) # Down
            grid[-1][i].obs = True
            grid[i][-1].show(BORDER, 0) # Left
            grid[i][-1].obs = True
            grid[i][0].show(BORDER, 0) # Right
            grid[i][0].obs = True


    if CUSTOM:
        start, end = None, None
        for r in range(ROWS):
            for c in range(COLS):
                if CUSTOM_GRID[c][r] == "S": 
                    start = grid[r][c]
                if CUSTOM_GRID[c][r] == "E": 
                    end = grid[r][c]
        if not start:
            for r in range(ROWS):
                for c in range(COLS):
                    if CUSTOM_GRID[c][r] == ".": 
                        start = grid[r][c]
                        break
        if not end:
            for r in range(ROWS-1,-1,-1):
                for c in range(COLS-1,-1,-1):
                    if CUSTOM_GRID[c][r] == ".": 
                        end = grid[r][c]
                        break
    else:
        start, end = grid[ST[0]][ST[1]], grid[ED[0]][ED[1]]

    pygame.init()

    start.show(START, 0)
    end.show(END, 0)

    def draw(pos):
        x, y = pos
        square = grid[x // w][y // h]
        if not square.obs and square != start and square != end:
            square.obs = True
            square.show(WALLS, 0)

    def erase(pos):
        x, y = pos
        square = grid[x // w][y // h]
        if square.obs and square != start and square != end:
            square.obs = False
            square.show(BG, 0)
            if CB: square.show(WALLS, 1)

    def weight(pos, n):
        x, y = pos
        square = grid[x // w][y // h]
        if not square.obs and square != start and square != end:
            if n == 1:
                square.value = 1
                square.show(BG, 0)
                if CB: square.show(WALLS, 1)
            elif n == 0:
                square.value = 0
                square.show(BG, 0)
                if not CB: square.show(WALLS, 1)
            elif n > 0:
                square.value = n
                square.show((15*n, 0, 0), 0)
                square.show((255, 0, 0), 1)
            else:
                square.value = n
                square.show((0, 15*(-n), 0), 0)
                square.show((0, 255, 0), 1)

    def reposition(item, pos):
        nonlocal start,end
        x, y = pos
        i, j = x // w, y // h
        if item == "start":
            if grid[i][j] == end or grid[i][j].obs: return
            start.show(BG,0)
            if CB: start.show(WALLS, 1)
            start = grid[i][j]
            start.show(START,0)
        else: 
            if grid[i][j] == start or grid[i][j].obs: return
            end.show(BG,0)
            if CB: end.show(WALLS, 1)
            end = grid[i][j]
            end.show(END,0)
    
    def clear_board():
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c].obs:
                    grid[r][c].obs = False
        screen.fill(BG)
        start.show(START, 0)
        end.show(END, 0)
        print("Cleared grid.")

    if CUSTOM:
        for r in range(ROWS):
            for c in range(COLS):
                if CUSTOM_GRID[c][r] == "#": draw((r*w+1, c*h+1))

    # Drawing time:
    drawing = True
    neg = 1 # Negative weights
    modeselect = False
    saving = False
    number_keys = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
                pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9}
    print("""Keybinds:
    Left Click: Draw walls | Right Click: Erase walls | Space: Begin pathfinding
    Number keys: Increase cell weights | N: Toggle negative weights
    P: Toggle show process | S/E: Reposition Start/End | M + Numbers: Change movement type
    X: Save board | C: Clear grid | D: Change cost calculation (Manhattan / Euclidean)""")
    while drawing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                drawing = False
                break
            if pygame.mouse.get_pressed()[0]: # Left click
                pos = pygame.mouse.get_pos()
                try:
                    draw(pos)
                except (AttributeError,IndexError):
                    continue
            elif pygame.mouse.get_pressed()[2]: # Right click
                pos = pygame.mouse.get_pos()
                try: 
                    erase(pos)
                except (AttributeError,IndexError):
                    continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Finding path...\n")
                    drawing = False
                    break
                elif event.key == pygame.K_n:
                    print("Toggled [Negative weights]")
                    neg = -1 if neg == 1 else 1
                elif event.key == pygame.K_p:
                    SHOW = not SHOW
                    print("Toggled [Show Process]",f"{"ON" if SHOW else "OFF"}")
                elif event.key == pygame.K_s:
                    pos = pygame.mouse.get_pos()
                    reposition("start", pos)
                elif event.key == pygame.K_e:
                    pos = pygame.mouse.get_pos()
                    reposition("end", pos)
                elif event.key == pygame.K_m:
                    modeselect = not modeselect
                    print("Selecting Mode",f"{"[ON]" if modeselect else "[OFF]"}")
                elif event.key == pygame.K_c:
                    clear_board()
                elif event.key == pygame.K_d:
                    DISTANCE = not DISTANCE
                    print(f"Pathfinding with {"Manhattan" if DISTANCE else "Euclidean"} distance.")
                elif event.key == pygame.K_t:
                    TIMING = not TIMING
                    print(f"Toggled stopwatch {"[ON]" if TIMING else "[OFF]"}")
                elif event.key == pygame.K_x:
                    saving = True
                    print("Current board:\n")
                    res = [[0 for _ in range(COLS)] for _ in range(ROWS)]
                    for r in range(ROWS):
                        for c in range(COLS):
                            res[r][c] = START_S if start == grid[c][r] else END_S if end==grid[c][r] else OBS_S if grid[c][r].obs else EMPTY_S
                    for line in res: print("".join(line))

                elif event.key in number_keys:
                    if modeselect:
                        MODE = number_keys[event.key]
                        print("Movement type:", modes[MODE])
                    else:
                        pos = pygame.mouse.get_pos()
                        try:
                            weight(pos, number_keys[event.key] * neg)
                        except (AttributeError,IndexError):
                            continue

    toCheck.append((0, (start.i,start.j)))

    # Add neighbors considering new obstacles
    for i in range(COLS):
        for j in range(ROWS):
            grid[i][j].addNeighbors(grid)

    # A* Heuristic function
    def heuristic(node, end):
        return abs(node.i - end.i) + abs(node.j - end.j) # Manhattan
    def euclidean(node, end):
        return sqrt((node.i - end.i)**2 + (node.j - end.j)**2)
    start.g = 0
    start.f = heuristic(start, end) if DISTANCE else euclidean(start, end)

    # Pathfinding
    def main():
        global repeat, TIMING
        """A* Pathfinding algorithm. You can draw obstacles on the screen, or increase the weight of squares using numbers.
        Press N to make negative weights. Press S to view the pathfinding process. Press Space to begin pathfinding."""
        if toCheck: # Check while nodes haven't been checked yet
            _, (x,y) = heapq.heappop(toCheck)
            current = grid[x][y]
            if current == end:
                distance = int(current.f)
                print("Path found. Distance:", distance)
                if TIMING: print(f"{perf()-starttime:.3f}s")
                while current != start: # Trace back previous path
                    current.processed = False
                    current.show(PATH, 0)
                    if saving and current != end: res[current.j][current.i] = PATH_S
                    current = current.previous
                end.show(END, 0)
                if saving:
                    print("Path:\n")
                    for line in res: print("".join(line))
                if messagebox.askokcancel("Path found", (f"A path has been found.\nThe lowest distance to the path is {str(distance//1)} blocks away.\nPress OK to try again, or cancel to exit.")):
                    repeat = True
                pygame.quit()
                return

            processed.add((x,y))
            if SHOW and grid[x][y] != start: grid[x][y].show(PROCESSED, 0)

            neighbors = current.neighbors
            for neighbor in neighbors: # Add neighbors
                a,b = neighbor.i, neighbor.j
                if (a,b) not in processed:
                    tempG = current.g + current.value
                    if tempG < neighbor.g:
                        neighbor.g = tempG
                        neighbor.h = heuristic(neighbor, end) if DISTANCE else euclidean(neighbor, end)
                        neighbor.f = neighbor.g + neighbor.h
                        neighbor.previous = current
                        heapq.heappush(toCheck, (neighbor.f, (a, b)))
                        if SHOW: neighbor.show(TOCHECK, 0)

        if not toCheck:
            print("There is no path to the end node.")
            pygame.quit()
            return
        current.processed = True
    
    if TIMING: starttime = perf()
    while True:
        if not pygame.get_init():
            break
        ev = pygame.event.poll()
        if ev.type == pygame.QUIT:
            pygame.quit()
            break
        pygame.display.update()
        main()

repeat = True
while repeat:
    repeat = False
    mainloop()
