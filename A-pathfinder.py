import pygame
import math
from tkinter import messagebox
import random

#? A* Pathfinding algorithm
# Unlike its predecessor Dijkstra, which only looks at the shortest path with a priority queue, this takes into
# consideration the distance from the end as well, with a heuristic estimate (Euclidean/Manhattan distance).
# The total cost thus is f = g + h, then it works the same as Dijkstra's algorithm with a priority queue

def mainloop():
    SHOW = False
    CUSTOM_GRID = None
    CUSTOM = bool(CUSTOM_GRID)

    ST, ED = (0,0), (-1,-1)
    ROWS, COLS = 30, 30
    MODE = 0
    modes = ["normal", "king", "horse", "diagonals", "jumper", "flash", "wallhugger","drunk","mirror", "wormhole"]
    border = False

    BG = (0,7,20)
    CB = False # Cell Borders
    WALLS = (220,227,213)
    TOCHECK = (30,175,14)
    PROCESSED = (200,0,0)
    PATH = (0,160,245)
    START = "green"
    END = "magenta"
    BORDER = "gray80"
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

    class Node:
        def __init__(self, x, y):
            self.i = x
            self.j = y
            self.f = 0 # Cost function: f = g + h
            self.g = 0 # Cost from start node to current
            self.h = 0 # Heuristic estimate from current to end node
            self.neighbors = []
            self.previous = None
            self.obs = False # Obstacle
            self.processed = False
            self.value = 1 # Default cost value for g

        def show(self, color, border):
            if self.processed == False:
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
                    dirs = [(random.choice([-1, 0, 1]), random.choice([-1, 0, 1])) for _ in range(8)]
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


    toCheck = [] # Nodes to be evaluated
    processed = [] # Evaluated nodes

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
                if CUSTOM_GRID[c][r] == "S": start = grid[r][c]
                if CUSTOM_GRID[c][r] == "E": end = grid[r][c]
        if not start:
            for r in range(ROWS):
                for c in range(COLS):
                    if CUSTOM_GRID[r][c] == ".": 
                        start = grid[r][c]
                        break
        if not end:
            for r in range(ROWS):
                for c in range(COLS):
                    if CUSTOM_GRID[r][c] == ".": 
                        end = grid[r][c]
                        break
    else:
        start, end = grid[ST[0]][ST[1]], grid[ED[0]][ED[1]]

    pygame.init()

    start.show(START, 0)
    end.show(END, 0)

    def draw(pos):
        x, y = pos[0], pos[1]
        i, j = x // w, y // h
        square = grid[i][j]
        if not square.obs and square != start and square != end:
            square.obs = True
            square.show(WALLS, 0)

    def erase(pos):
        x, y = pos[0], pos[1]
        i, j = x // w, y // h
        square = grid[i][j]
        if square.obs and square != start and square != end:
            square.obs = False
            square.show(BG, 0)
            if CB: square.show(WALLS, 1)

    def weight(pos, n):
        x, y = pos[0], pos[1]
        i, j = x // w, y // h
        square = grid[i][j]
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
        global start,end
        x, y = pos[0], pos[1]
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
    print("""\nLeft Click: Draw walls | Right Click: Erase walls | Space: Begin pathfinding
    Numbers: Increase cell weights | N: Toggle negative weights\nP: Toggle show progress | S/E: Reposition Start/End | M + Numbers: Change movement type\nX: Save board""")
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
                elif event.key == pygame.K_x:
                    saving = True
                    print("Current board:\n")
                    res = [[0 for _ in range(COLS)] for _ in range(ROWS)]
                    for r in range(ROWS):
                        for c in range(COLS):
                            try: res[r][c] = START_S if start == grid[c][r] else END_S if end==grid[c][r] else OBS_S if grid[c][r].obs else EMPTY_S
                            except: print(r,c)
                    print("[")
                    for line in res: print(line,",",sep="")
                    print("]")

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

    toCheck.append(start)

    # Add neighbors considering new obstacles
    for i in range(COLS):
        for j in range(ROWS):
            grid[i][j].addNeighbors(grid)

    # A* Heuristic function
    def heurisitic(node, end):
        return math.sqrt((node.i - end.i)**2 + (node.j - end.j)**2)

    # Pathfinding
    def main():
        global repeat
        """A* Pathfinding algorithm. You can draw obstacles on the screen, or increase the weight of squares using numbers.
        Press N to make negative weights. Press S to view the pathfinding process. Press Space to begin pathfinding."""
        start.show(START, 0)
        end.show(END, 0)
        if toCheck: # Check while nodes haven't been checked yet
            lowestIndex = 0
            for i in range(len(toCheck)): # Find lowest f score
                if toCheck[i].f < toCheck[lowestIndex].f:
                    lowestIndex = i

            current = toCheck[lowestIndex]
            if current == end:
                distance = int(current.f)
                print("Path found. Distance:", distance)
                start.show(START,0)
                while current.previous: # Trace back previous path
                    current.processed = False
                    current.show(PATH, 0)
                    if saving and current != end: res[current.j][current.i] = PATH_S
                    current = current.previous
                end.show(END, 0)

                if saving: 
                    print("Path:\n")
                    for line in res: print(line,",",sep="")

                if messagebox.askokcancel("Path found", ("A path has been found.\nThe lowest distance to the path is " + str(distance//1) + " blocks away.\nPress OK to try again, or cancel to exit.")):
                    repeat = True
                pygame.quit()
                return

            toCheck.pop(lowestIndex)
            processed.append(current)

            neighbors = current.neighbors
            for i in range(len(neighbors)): # Add neighbors
                neighbor = neighbors[i]
                if neighbor not in processed:
                    tempG = current.g + current.value
                    if neighbor in toCheck:
                        if neighbor.g > tempG: # If this cost is smaller than before, you set it as the new cost (we found a shortest path to the node)
                            neighbor.g = tempG
                            neighbor.previous = current 
                    else:
                        neighbor.g = tempG
                        toCheck.append(neighbor)
                        neighbor.previous = current 

                neighbor.h = heurisitic(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h
                
        if SHOW:
            for i in range(len(toCheck)):
                toCheck[i].show(TOCHECK, 0)

            for i in range(len(processed)):
                if processed[i] != start:
                    processed[i].show(PROCESSED, 0)
        
        if not toCheck:
            print("There is no path to the end node.")
            pygame.quit()
            return
        current.processed = True

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
