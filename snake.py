import pygame
import time
import random
from collections import deque
pygame.font.init()

WIDTH, HEIGHT = 800, 800
N = 20
w, h = WIDTH // N, HEIGHT // N
POSX, POSY = 0, 0
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")
WALL = "white"
SNAKE = "lime"
FOOD = "red"
BG = "black"

INIT_SPEED = 1
START_DIR = -1

class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.wall = False
        self.snake = False
        self.food = False
        self.dir = 0
    
    def draw(self, color = None):
        if self.snake:
            pygame.draw.rect(win, SNAKE, (POSX + self.x * w, POSY + self.y * h, w, h))
        elif self.food:
            pygame.draw.rect(win, FOOD, (POSX + self.x * w, POSY + self.y * h, w, h))
        elif self.wall:
            pygame.draw.rect(win, WALL, (POSX + self.x * w, POSY + self.y * h, w, h))
        else:
            pygame.draw.rect(win, BG, (POSX + self.x * w, POSY + self.y * h, w, h))

class Board:
    def __init__(self):
        self.grid = self.makegrid()
        self.snake = None
        self.food = 1

    def makegrid(self):
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

    def spawn_food(self):
        options = [self.grid[i][j] for i in range(N) for j in range(N) if not self.grid[i][j].snake and not self.grid[i][j].wall]
        for _ in range(self.food):
            x = random.choice(options)
            options.remove(x)
            x.food = True
            x.draw()

class Snake:
    def __init__(self, board, x, y):
        self.board = board
        self.head = board.grid[x][y]
        self.head.snake = True
        self.head.dir = START_DIR
        self.tail = self.head
        self.size = 1
        self.speed = INIT_SPEED
        self.body = deque([self.head])
        self.direct = START_DIR
        self.directs = {1: (0,1), -1: (0,-1), 2: (1,0), -2: (-1,0)}

    def dirchange(self, direct = START_DIR):
        if direct != -self.direct:
            self.direct = direct

    def move(self):
        x,y = self.directs[self.direct]
        next = (self.head.x + x, self.head.y + y)
        if not (0<=next[0]<N and 0<=next[1]<N): return False
        next = self.board.grid[self.head.x + x][self.head.y + y]
        if next.snake or next.wall:
            return False
        elif next.food:
            self.eat(next)
        
        # Tail movement
        t = self.body.popleft()
        t.snake = False
        t.dir = 0
        t.draw()

        # Head movement
        self.body.append(next)
        next.snake = True
        next.dir = self.direct

        self.head = self.body[-1]
        self.tail = self.body[0]
        self.tail.draw()
        self.head.draw()
        return True

    def eat(self, food):
        food.food = False
        self.board.spawn_food()
        self.size += 1
        old = self.body[0]
        x,y = old.x, old.y
        match old.dir:
            case 1:
                new = self.board.grid[x][y-1]
            case -1:
                new = self.board.grid[x][y+1]
            case 2:
                new = self.board.grid[x-1][y]
            case -2:
                new = self.board.grid[x+1][y]
        self.body.appendleft(new)
        new.snake = True
        new.draw()


def main():
    board = Board()
    snake = Snake(board, N//2, N//2)
    board.snake = snake
    clock = pygame.time.Clock()
    run = True
    board.spawn_food()
    board.draw()
    while run:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    snake.dirchange(2)
                elif event.key == pygame.K_LEFT:
                    snake.dirchange(-2)
                elif event.key == pygame.K_DOWN:
                    snake.dirchange(1)
                elif event.key == pygame.K_UP:
                    snake.dirchange(-1)
        if not snake.move():
            run = False
            break
        pygame.display.update()
    print("Game over!")
    pygame.quit()
    return


main()
