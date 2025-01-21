import pygame
import random
from collections import deque
from time import sleep
pygame.font.init()
pygame.mixer.init()

NOM = pygame.mixer.Sound("./sounds/nom.mp3")

N = 16
WIDTH, HEIGHT = 800, 800
WIDTH, HEIGHT = WIDTH + (N - WIDTH % N), HEIGHT + (N - HEIGHT % N) # Fix borders
w, h = WIDTH // N, HEIGHT // N
POSX, POSY = 0, 0
win = None # Window, initialized later
SNAKEBORDERCOLOR = SNAKEBORDER = "#12653e"
BORDER = True
BORDERSIZE = 1 if N > 60 else 2 if N > 35 else 3
SNAKECOLOR = SNAKE = "#42f54e"
WINCOLOR = "#fcba03"
WINBORDER = "#ad7f00"
LOSECOLOR = "#700000"
LOSEBORDER = "#240000"
INVINCIBILITY = "white"
INVINCIBILITY_BORDER = "#bababa"
FOOD = "red"
BG = "black"
FONT = pygame.font.SysFont("comicsans", 20)
FONTCOLOR = "white"
SOUND = True

ADMIN = True
START_DIR = -1
PORTALS = True
DOUBLEFOOD = 50 + N
TRIPLEFOOD = 150 + N
WIN_REQ = N*N-1

class Cell:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y
        self.snake = False
        self.food = False
        self.dir = 0
    
    def draw(self, color: str = None, border: int = 0):
        if not color:
            if self.snake:
                pygame.draw.rect(win, SNAKE, (POSX + self.x * w, POSY + self.y * h, w, h))
                BORDER and pygame.draw.rect(
                    win, SNAKEBORDER, (POSX + self.x * w, POSY + self.y * h, w, h), BORDERSIZE
                    )
            elif self.food:
                pygame.draw.rect(win, FOOD, (POSX + self.x * w, POSY + self.y * h, w, h))
            else:
                pygame.draw.rect(win, BG, (POSX + self.x * w, POSY + self.y * h, w, h))
        else:
            pygame.draw.rect(win, color, (POSX + self.x * w, POSY + self.y * h, w, h), border)


class Board:
    def __init__(self):
        self.grid = self.makegrid()
        self.snake = None
        self.food = 1
        self.currentfood = 0

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

    def spawn_food(self):
        if self.currentfood < 0: self.currentfood = 0
        options = []
        for i in range(N):
            for j in range(N):
                if not self.grid[i][j].snake and not self.grid[i][j].food: 
                    options.append(self.grid[i][j])
        for _ in range(self.food-self.currentfood):
            try: x = random.choice(options)
            except: 
                if ADMIN: print("ADMIN log: Error - unable to spawn food.")
                return
            options.remove(x)
            x.food = True
            x.draw()
        self.currentfood = self.food
    def update_score(self, score: int):
        a = 4 if N >= 4 else N
        b = 2 if N >= 2 else N
        if N > 25:
            b = N//10
            a = N//5
        for i in range(a):
            for j in range(b):
                self.grid[i][j].draw()
        sc = FONT.render(f"Score: {str(score)}", True, FONTCOLOR)
        win.blit(sc, (5, 5))

class Snake:
    def __init__(self, board: Board, x: int , y: int):
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

    def dirchange(self, direct: int = START_DIR) -> None:
        if direct != -self.direct:
            self.direct = direct

    def move(self) -> bool:
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
            if SOUND: self.eat(next)
            else: self.eat(next,sound=False)

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

    def eat(self, food: Cell, cheat: bool = False, sound = True) -> None:
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
        if sound: pygame.mixer.Sound.play(NOM)
        if self.size == DOUBLEFOOD : self.board.food += 1
        if self.size == TRIPLEFOOD: self.board.food += 1

class Game:
    def __init__(self):
        global win
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")
        self.board = Board()
        self.snake = Snake(self.board, N//2, N//2)
        self.board.snake = self.snake
        self.clock = pygame.time.Clock()
        self.run = True
        self.paused = False
        self.speed = 10 if N > 20 else 8 if N > 10 else 6
        self.score = 0
        self.board.spawn_food()

    def restart(self):
        self.__init__()

    def main_loop(self):
        global BORDER, PORTALS,SOUND
        self.board.draw()
        self.board.update_score(1)
        if self.snake.inv:
            global SNAKE, SNAKEBORDER
            SNAKE = INVINCIBILITY
            SNAKEBORDER = INVINCIBILITY_BORDER
        while self.run:
            self.clock.tick(self.speed)
            moved = False
            self.score = self.snake.size
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and not self.paused:
                        # Move right
                        self.snake.dirchange(2)
                        stat = self.snake.move()
                        moved = True
                    elif (event.key == pygame.K_LEFT or event.key == pygame.K_a) and not self.paused:
                        # Move left
                        self.snake.dirchange(-2)
                        stat = self.snake.move()
                        moved = True
                    elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and not self.paused:
                        # Move down
                        self.snake.dirchange(1)
                        stat = self.snake.move()
                        moved = True
                    elif (event.key == pygame.K_UP or event.key == pygame.K_w) and not self.paused:
                        # Move up
                        self.snake.dirchange(-1)
                        stat = self.snake.move()
                        moved = True
                    elif event.key == pygame.K_SPACE and ADMIN:
                        # Toggle portals
                        PORTALS = not PORTALS
                        print(f"Portals toggled {'ON' if PORTALS else 'OFF'}")
                    elif event.key == pygame.K_i and ADMIN:
                        # Toggle invincibility
                        self.snake.inv = not self.snake.inv
                        SNAKE = INVINCIBILITY if self.snake.inv else SNAKECOLOR
                        SNAKEBORDER = INVINCIBILITY_BORDER if self.snake.inv else SNAKEBORDERCOLOR
                        if ADMIN: print(f"Invincibility toggled {'ON' if self.snake.inv else 'OFF'}")
                    elif event.key == pygame.K_k and ADMIN:
                        # Increase snake size
                        self.snake.eat(self.snake.head, True, False)
                    elif event.key == pygame.K_l and self.snake.size > 1 and ADMIN:
                        # Decrease snake size
                        self.snake.size -= 1
                        t = self.snake.body.popleft()
                        t.snake = False
                        t.dir = 0
                        t.draw()
                        self.snake.tail = self.snake.body[0]
                    elif event.key == pygame.K_m:
                        SOUND = not SOUND
                    elif event.key == pygame.K_o and ADMIN:
                        # Increase game speed
                        self.speed += 1
                    elif event.key == pygame.K_p and self.speed > 1 and ADMIN:
                        # Decrease game speed
                        self.speed -= 1
                    elif event.key == pygame.K_v and ADMIN:
                        # Check game speed
                        print(f"Current game speed: {self.speed}")
                    elif event.key == pygame.K_b and ADMIN:
                        # Toggle border for the snake
                        BORDER = not BORDER
                    elif event.key == pygame.K_b and ADMIN:
                        # Instant win
                        self.snake.size = WIN_REQ
                        self.snake.inv = True
                    elif event.key == pygame.K_r:
                        # Restart the game
                        self.restart()
                        if ADMIN: print("Restarted.")
                    elif event.key == pygame.K_ESCAPE:
                        # Pause
                        self.paused = not self.paused

            if not self.paused:
                if not moved:
                    stat = self.snake.move()
                if self.score == WIN_REQ:
                    self.score += 1
                    self.board.update_score(self.score)
                    for c in reversed(self.snake.body):
                        c.draw(WINCOLOR)
                        if BORDER: c.draw(WINBORDER, BORDERSIZE)
                        pygame.display.update()
                        sleep(round(3/len(self.snake.body), 2))
                        
                    self.run = False
                    break
                if not stat:
                    self.run = False
                    for c in reversed(self.snake.body):
                        c.draw(LOSECOLOR)
                        if BORDER: c.draw(LOSEBORDER, BORDERSIZE)
                        pygame.display.update()
                        sleep(round(1/len(self.snake.body), 2))
                    break
                self.board.update_score(self.score)
                pygame.display.update()
        print(f"Game over! Final score: {self.score}")
        if self.score >= WIN_REQ: print(f"Congratulations, you won in the {N}x{N} board!")
        pygame.quit()

def main():
    print("Keybinds: WASD (or arrow keys) - Move, R - Restart, ESC - Pause, M - Mute")
    if ADMIN:
        print("\tI: Invincibiility | K/L: +/-1 score | O/P: +/-1 speed | B: Instant win")
    game = Game()
    game.main_loop()

if __name__ == "__main__":
    main()
