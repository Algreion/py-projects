import pygame
import math
from tkinter import *
from tkinter import ttk
from tkinter import messagebox


# A* Pathfinding algorithm
# Unlike its predecessor Dijkstra, which only looks at the shortest path with a priority queue, this takes into
# consideration the distance from the end as well, with a heuristic estimate (Euclidean/Manhattan distance).
# The total cost thus is f = g + h, then it works the same as Dijkstra's algorithm with a priority queue

# To add: 
# Actual priority queue (minHeap) to speed up pathfinding / other speed ups to make it faster
# Change colors
# Get rid of all Tkinter stuff (terminal interface?)
# Refactor code (Encapsulate, wrap up in classes, avoid spilling outside __main__)
# Make a better end screen (both for finding and not finding path)
# Remove window border?
# Scale down cell border if lots of cells (also fix width/height problem with 100+ cells)



ASK = False
SHOW = False

ST, ED = (0,0), (-1,-1)
COLS, ROWS = 20, 20
border = False

if border:
    COLS += 2
    ROWS += 2
    if ST == (0, 0): ST = (1,1)
    if ED == (-1, -1): ED = (-2, -2)

grid = [0 for _ in range(COLS)] # Empty array
toCheck = [] # Nodes to be evaluated
processed = [] # Evaluated nodes
TOCHECK = (0,255,0)
PROCESSED = (255,0,0)
PATH = (0,0,255)
START = "orange"
END = "magenta"
BORDER = "gray80"
WIDTH = 800
HEIGHT = 800
w = WIDTH // COLS # Width of nodes
h = HEIGHT // ROWS # Height of nodes


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding")

class Node:
    def __init__(self, x, y):
        self.i = x
        self.j = y
        self.f = 0 # Cost function: f = g + h
        self.g = 0 # Cost from start node to current
        self.h = 0 # Heuristic estimate from current to end node
        self.neighbors = [] # Adjacent path nodes
        self.previous = None # Previous node in path
        self.obs = False # Checks if it's an obstacle
        self.processed = False # Checks if node has been evaluated
        self.value = 1 # Default value for g (It costs 1 to travel to any adjacent neighbor)

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
        # If not obstacle, add as neighbor
        if i < COLS-1 and not grid[self.i + 1][j].obs: 
            self.neighbors.append(grid[self.i + 1][j])
        if i > 0 and not grid[self.i - 1][j].obs:
            self.neighbors.append(grid[self.i - 1][j])
        if j < ROWS-1 and not grid[self.i][j + 1].obs:
            self.neighbors.append(grid[self.i][j + 1])
        if j > 0 and not grid[self.i][j - 1].obs:
            self.neighbors.append(grid[self.i][j - 1])


# Create 2d array
for i in range(COLS):
    grid[i] = [0 for _ in range(ROWS)]

# Create nodes
for i in range(COLS):
    for j in range(ROWS):
        grid[i][j] = Node(i, j)

# Show the empty grid
for i in range(COLS):
    for j in range(ROWS):
        grid[i][j].show((255, 255, 255), 1)

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




if ASK:
    # Tkinter box
    def onsubmit():
        global start
        global end
        st = startBox.get().split(',')
        ed = endBox.get().split(',')
        start = grid[int(st[0])][int(st[1])]
        end = grid[int(ed[0])][int(ed[1])]
        window.quit()
        window.destroy()

    window = Tk()
    label = Label(window, text='Start (x,y): ')
    startBox = Entry(window)
    label1 = Label(window, text='End (x,y): ')
    endBox = Entry(window)
    var = IntVar()
    showPath = ttk.Checkbutton(window, text='Show Steps :', onvalue=1, offvalue=0, variable=var)

    submit = Button(window, text='Submit', command=onsubmit)

    label.grid(row=0, pady=3)
    label1.grid(row=1, pady=3)
    startBox.grid(row=0, column=1, pady=3)
    endBox.grid(row=1, column=1, pady=3)
    showPath.grid(columnspan=2, row=2)
    submit.grid(columnspan=2, row=3)


    window.update()
    mainloop() # Mainloop of Tkinter
else:
    start, end = grid[ST[0]][ST[1]], grid[ED[0]][ED[1]]

pygame.init()
toCheck.append(start)

start.show(START, 0)
end.show(END, 0)

def draw(pos):
    x, y = pos[0], pos[1]
    i, j = x // w, y // h
    square = grid[i][j]
    if not square.obs and square != start and square != end:
        square.obs = True
        square.show((255, 255, 255), 0)
def erase(pos):
    x, y = pos[0], pos[1]
    i, j = x // w, y // h
    square = grid[i][j]
    if square.obs and square != start and square != end:
        square.obs = False
        square.show((0, 0, 0), 0)
        square.show((255, 255, 255), 1)

def weight(pos, n):
    x, y = pos[0], pos[1]
    i, j = x // w, y // h
    square = grid[i][j]
    if not square.obs and square != start and square != end:
        if n == 1:
            square.value = 1
            square.show((0, 0, 0), 0)
            square.show((255, 255, 255), 1)
        elif n == 0:
            square.value = 0
            square.show((0, 0, 0), 0)
        elif n > 0:
            square.value = n
            square.show((15*n, 0, 0), 0)
            square.show((255, 0, 0), 1)
        else:
            square.value = n
            square.show((0, 15*(-n), 0), 1)
            square.show((0, 255, 0), 1)


# Drawing time:
drawing = True
neg = 1 # Negative weights
number_keys = {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
               pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9}
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
                print("Finding path...")
                drawing = False
                break
            elif event.key == pygame.K_n:
                print("Toggled [Negative weights]")
                neg = -1 if neg == 1 else 1
            elif event.key == pygame.K_s:
                SHOW = False if SHOW else True
                print("Toggled [Show Process]")
            elif event.key in number_keys:
                pos = pygame.mouse.get_pos()
                try:
                    weight(pos, number_keys[event.key] * neg)
                except (AttributeError,IndexError):
                    continue

# Add neighbors considering new obstacles
for i in range(COLS):
    for j in range(ROWS):
        grid[i][j].addNeighbors(grid)

# A* Heuristic function
def heurisitic(node, end):
    d = math.sqrt((node.i - end.i)**2 + (node.j - end.j)**2)
    # d = Estimated cost between the currend node and the end position
    return d

# Pathfinding
def main():
    """A* Pathfinding algorithm. You can draw obstacles on the screen, or increase the weight of squares using numbers. 
    Press N to make negative weights. Press S to view the pathfinding process. Press Space to begin pathfinding."""
    start.show(START, 0)
    end.show(END, 0)
    if toCheck: # Check while nodes haven't been checked yet
        lowestIndex = 0
        for i in range(len(toCheck)): # Find lowest f score (slowest part, need a minHeap here)
            if toCheck[i].f < toCheck[lowestIndex].f:
                lowestIndex = i

        current = toCheck[lowestIndex] # Set current to it
        if current == end: # If it's the end:
            distance = int(current.f)
            print("Path found. Distance:", distance)
            start.show(START,0)
            while current.previous: # Trace back previous path with node.previous
                current.processed = False
                current.show(PATH, 0)
                current = current.previous
            end.show(END, 0)

            Tk().wm_withdraw()
            result = messagebox.askokcancel("Path found", ("A path has been found.\nThe lowest distance to the path is " + str(distance//1) + " blocks away.\nPress OK to close."))
            if result or not result:
                pygame.quit()
                return
            
        toCheck.pop(lowestIndex)
        processed.append(current) # We've checked that node

        neighbors = current.neighbors
        for i in range(len(neighbors)): # Add neighbors
            neighbor = neighbors[i]
            if neighbor not in processed:
                tempG = current.g + current.value # Evaluate current cost value
                if neighbor in toCheck:
                    if neighbor.g > tempG: # If this cost is smaller than before, you set it as the new cost (we found a shortest path to the node)
                        neighbor.g = tempG
                        neighbor.previous = current 
                else:
                    neighbor.g = tempG
                    toCheck.append(neighbor) # Add in toCheck
                    neighbor.previous = current 

            neighbor.h = heurisitic(neighbor, end)
            neighbor.f = neighbor.g + neighbor.h
            
    if SHOW or ASK and var.get():
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


# Code for moving in tiles
def update(self, tilemap, movement=(0, 0)):
    frame_movement = (movement[0] + self.velocity[0], movement[1] + self.velocity[1])

    self.pos[0] += frame_movement[0]
    entity_rect = self.rect()
    for rect in tilemap.physics_rects_around(self.pos):
        if entity_rect.colliderect(rect):
            if frame_movement[0] > 0:
                entity_rect.right = rect.left
            else:
                entity_rect.left = rect.right
    
    self.pos[1] += frame_movement[1]
    self.velocity[1] = min(5, self.velocity[1] + 0.1)