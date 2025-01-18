import pygame
import random
from collections import deque
from time import sleep
pygame.font.init()

N = 15
WIDTH, HEIGHT = 800 + (N - 800%N), 800 + (N - 800%N)
w, h = WIDTH // N, HEIGHT // N
POSX, POSY = 0, 0
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")
SNAKEBORDER = "#12653e"
BORDER = True
SNAKE = "#42f54e"
FOOD = "red"
BG = "black"
FONT = pygame.font.SysFont("comicsans", 20)
FONTCOLOR = "white"

START_DIR = -1
PORTALS = True
ADMIN = True
DOUBLEFOOD = 15 + N
TRIPLEFOOD = 100 + N

class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.snake = False
        self.food = False
        self.dir = 0
    
    def draw(self, color = None):
        if not color:
            if self.snake:
                pygame.draw.rect(win, SNAKE, (POSX + self.x * w, POSY + self.y * h, w, h))
                BORDER and pygame.draw.rect(win, SNAKEBORDER, (POSX + self.x * w, POSY + self.y * h, w, h), 3)
            elif self.food:
                pygame.draw.rect(win, FOOD, (POSX + self.x * w, POSY + self.y * h, w, h))
            else:
                pygame.draw.rect(win, BG, (POSX + self.x * w, POSY + self.y * h, w, h))
        else:
            pygame.draw.rect(win, color, (POSX + self.x * w, POSY + self.y * h, w, h))

class Board:
    def __init__(self):
        self.grid = self.makegrid()
        self.snake = None
        self.food = 1
        self.currentfood = 0

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
        if self.currentfood < 0: self.currentfood = 0
        options = []
        for i in range(N):
            for j in range(N):
                if not self.grid[i][j].snake and not self.grid[i][j].food: options.append(self.grid[i][j])
        for _ in range(self.food-self.currentfood):
            try: x = random.choice(options)
            except: break
            options.remove(x)
            x.food = True
            x.draw()
        self.currentfood = self.food
    def update_score(self, score):
        a = 4 if N >= 4 else N
        b = 2 if N >= 2 else N
        for i in range(a):
            for j in range(b):
                self.grid[i][j].draw()
        sc = FONT.render(f"Score: {str(score)}", True, FONTCOLOR)
        win.blit(sc, (5, 5))

class Snake:
    def __init__(self, board, x, y):
        self.board = board
        self.head = board.grid[x][y]
        self.head.snake = True
        self.head.dir = START_DIR
        self.tail = self.head
        self.size = 1
        self.speed = 1
        self.body = deque([self.head])
        self.direct = START_DIR
        self.directs = {1: (0,1), -1: (0,-1), 2: (1,0), -2: (-1,0)}
        self.inv = False

    def dirchange(self, direct = START_DIR):
        if direct != -self.direct:
            self.direct = direct

    def move(self):
        x,y = self.directs[self.direct]
        next_x, next_y = self.head.x + x, self.head.y + y
        if not (0 <= next_x < N and 0 <= next_y < N):
            if PORTALS or self.inv:
                if next_x < 0: next_x = N - 1
                if next_y < 0: next_y = N - 1
                if next_x >= N: next_x = 0
                if next_y >= N: next_y = 0
            else: return False
        next = self.board.grid[next_x][next_y]

        if next.snake and not self.inv:
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

    def eat(self, food, cheat = False):
        if not cheat:
            food.food = False
            self.board.currentfood -= 1
            self.board.spawn_food()
        self.size += 1
        old = self.body[0]
        x,y = old.x, old.y
        match old.dir:
            case 1:
                x,y = x,y-1
            case -1:
                x,y = x,y+1
            case 2:
                x,y = x-1,y
            case -2:
                x,y = x+1,y
        try:
            new = self.board.grid[x][y]
        except:
            self.size -= 1
            if ADMIN: print(f"ADMIN log: Error - unable to create new snake body at position x: {x}, y: {y}.")
            return
        self.body.appendleft(new)
        new.snake = True
        new.draw()
        if self.size == DOUBLEFOOD : self.board.food += 1
        if self.size == TRIPLEFOOD: self.board.food += 1


def main():
    global PORTALS, BORDER
    board = Board()
    snake = Snake(board, N//2, N//2)
    board.snake = snake
    clock = pygame.time.Clock()
    run = True
    board.spawn_food()
    board.draw()
    speed = 10 if N > 20 else 8 if N > 10 else 6
    board.update_score(1)
    score = 0
    while run:
        clock.tick(speed)
        moved = False
        score = snake.size
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    snake.dirchange(2)
                    stat = snake.move()
                    moved = True
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    snake.dirchange(-2)
                    stat = snake.move()
                    moved = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    snake.dirchange(1)
                    stat = snake.move()
                    moved = True
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    snake.dirchange(-1)
                    stat = snake.move()
                    moved = True
                elif event.key == pygame.K_SPACE and ADMIN:
                    PORTALS = not PORTALS
                    print(f"Portals toggled {"ON" if PORTALS else "OFF"}")
                elif event.key == pygame.K_i and ADMIN:
                    snake.inv = not snake.inv
                    print(f"Invincibility toggled {"ON" if snake.inv else "OFF"}")
                elif event.key == pygame.K_k and ADMIN:
                    snake.eat(snake.head, True)
                elif event.key == pygame.K_l and snake.size > 1 and ADMIN:
                    snake.size -= 1
                    t = snake.body.popleft()
                    t.snake = False
                    t.dir = 0
                    t.draw()
                    snake.tail = snake.body[0]
                elif event.key == pygame.K_o and ADMIN:
                    speed += 1
                elif event.key == pygame.K_p and speed > 1 and ADMIN:
                    speed -= 1
                elif event.key == pygame.K_b and ADMIN:
                    BORDER = not BORDER

        if not moved:
            stat = snake.move()
        if not stat:
            run = False
            break
        board.update_score(score)
        if score == N*N:
            print("Congratulations, you won!")
            for c in reversed(snake.body):
                c.draw("white")
                pygame.display.update()
                sleep(0.05)
            run = False
            break
        pygame.display.update()
    print(f"Game over! Final score: {score}")
    pygame.quit()
    return


main()
