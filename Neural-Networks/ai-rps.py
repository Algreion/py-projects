from collections import deque
from micrograd import *
import pygame
pygame.font.init()

ACTIVATION = 5
LAYERS = [8,8] # Hidden layers
LOG = False
CUSTOM = True
CUSTOM_PARAMS = 'rps-model.txt'

WIDTH, HEIGHT = 900,500
BUTTON_FONTSIZE = 40
BG_COLOR = (240,240,240)
FONT_COLOR = (0,0,0)
BUTTON_FONT_COLOR = (255,255,255)
WINNING_COLOR = (0,190,0)
LOSING_COLOR = (190,0,0)
BUTTON_BORDER_COLOR = (0,0,0)
BUTTON_BORDER_SIZE = 3
ROCK_COLOR = (120,120,120)
PAPER_COLOR = (90,90,90)
SCISSOR_COLOR = (60,60,60)

BUTTONS_GAP = 25
BUTTONS_WIDTH = WIDTH//4
BUTTONS_HEIGHT = HEIGHT//4
BUTTONS_POSX = WIDTH//2 - 3*BUTTONS_WIDTH//2 - BUTTONS_GAP
BUTTONS_POSY = HEIGHT-20

ROCK_BUTTON = (BUTTONS_POSX, BUTTONS_POSY - BUTTONS_HEIGHT, BUTTONS_WIDTH, BUTTONS_HEIGHT) # x, y, width, height
PAPER_BUTTON = (BUTTONS_POSX + BUTTONS_GAP + BUTTONS_WIDTH, BUTTONS_POSY - BUTTONS_HEIGHT, BUTTONS_WIDTH, BUTTONS_HEIGHT)
SCISSORS_BUTTON = (BUTTONS_POSX + BUTTONS_GAP*2 + BUTTONS_WIDTH * 2, BUTTONS_POSY - BUTTONS_HEIGHT,  BUTTONS_WIDTH, BUTTONS_HEIGHT)

SCORES_FONTSIZE = 25
PLAYER_SCORE_POSX, PLAYER_SCORE_POSY = WIDTH-20, 10
AI_SCORE_POSX, AI_SCORE_POSY = WIDTH-20, 40

MSG_FONTSIZE = 30
MSG_POSX, MSG_POSY = WIDTH//2, HEIGHT//2 - 100

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors")
FONT = pygame.font.SysFont("comicsans", BUTTON_FONTSIZE)
SCORESFONT = pygame.font.SysFont("comicsans", SCORES_FONTSIZE)
MSGFONT = pygame.font.SysFont("Verdana", MSG_FONTSIZE)

class Game:
    def __init__(self):
        self.ai = MLP(ACTIVATION + 3, LAYERS + [1])
        if CUSTOM:
            try:
                self.ai.load(CUSTOM_PARAMS)
            except:
                print("Failed to load custom model - defaulting.")
        self.sp = 0 # Player score
        self.sai = 0 # AI Score
        self.pattern = deque()
        self.prob = [0, 0, 0] # Probability counter
        self.counts = {-1: 0, 0: 0, 1: 0} # -1 Rock, 0 Paper, 1 Scissors
        self.rpstable = {-1: 1, 0: -1, 1: 0} # RPS table
        self.wintable = {-1: 0, 0: 1, 1: -1}
        self.total = 0 # Number of matches
        self.played = ""
        self.result = ""
        self.resultcolor = FONT_COLOR
        self.correct = 0
        self.prediction = None

    def handle_click(self, pos: tuple):
        x, y = pos
        if ROCK_BUTTON[0] <= x <= ROCK_BUTTON[0]+ROCK_BUTTON[2] and ROCK_BUTTON[1] <= y <= ROCK_BUTTON[1]+ROCK_BUTTON[3]:
            self.handle_input(-1)
        elif PAPER_BUTTON[0] <= x <= PAPER_BUTTON[0]+PAPER_BUTTON[2] and PAPER_BUTTON[1] <= y <= PAPER_BUTTON[1]+PAPER_BUTTON[3]:
            self.handle_input(0)
        elif SCISSORS_BUTTON[0] <= x <= SCISSORS_BUTTON[0]+SCISSORS_BUTTON[2] and SCISSORS_BUTTON[1] <= y <= SCISSORS_BUTTON[1]+SCISSORS_BUTTON[3]:
            self.handle_input(1)

    def handle_input(self, button: int):
        if self.total > ACTIVATION:
            inputs = list(self.pattern) + self.prob
            res = self.wintable[round(train(self.ai, inputs, button).data)]
            self.pattern.popleft()
            self.pattern.append(button)
        elif self.total == ACTIVATION:
            self.prob = [self.counts[-1]/self.total,self.counts[0]/self.total,self.counts[1]/self.total]
            inputs = list(self.pattern) + self.prob
            res = self.wintable[round(train(self.ai, inputs, button).data)]
            self.pattern.popleft()
            self.pattern.append(button)
        else:
            self.pattern.append(button)
            res = random.randint(-1,1)
        self.total += 1
        self.counts[button] += 1
        self.prob = [self.counts[-1]/self.total,self.counts[0]/self.total,self.counts[1]/self.total]
        self.handle_result(button, res)

    def handle_result(self, player: int, ai: int):
        self.played = f"Player: {"Rock" if player==-1 else "Paper" if player==0 else "Scissors"} | AI: {"Rock" if ai==-1 else "Paper" if ai==0 else "Scissors"}"
        if self.rpstable[player] == ai:
            self.sp += 1
            self.result = "Won!"
            self.resultcolor = WINNING_COLOR
        elif player == ai:
            self.result = "Tie!"
            self.resultcolor = FONT_COLOR
        else:
            self.sai += 1
            self.result = "Lost!"
            self.resultcolor = LOSING_COLOR
        
        if self.total > ACTIVATION:
            if player == self.prediction: self.correct += 1
            pygame.display.set_caption(f"Rock Paper Scissors | {self.total}. Accuracy: {str(round((self.correct/(self.total-ACTIVATION))*100,2))}%")
            if LOG: print(f"{self.total-1}. Player played: {'Rock' if player==-1 else 'Paper' if player==0 else 'Scissors'}")
        if self.total >= ACTIVATION:
            inp = list(self.pattern)+self.prob
            r = round(self.ai(inp).data)
            self.prediction = r
            if LOG: print(f"{self.total}. Prediction: {'Rock' if r==-1 else 'Paper' if r==0 else 'Scissors'}")
        self.draw()

    def draw(self):
        win.fill(BG_COLOR)
        # Rock Button
        pygame.draw.rect(win, ROCK_COLOR, ROCK_BUTTON)
        pygame.draw.rect(win, BUTTON_BORDER_COLOR, ROCK_BUTTON, BUTTON_BORDER_SIZE)
        txt = FONT.render("ROCK", True, BUTTON_FONT_COLOR)
        win.blit(txt, (BUTTONS_POSX + BUTTONS_WIDTH // 2 - txt.get_size()[0]//2,
                       BUTTONS_POSY - BUTTONS_HEIGHT // 2 - txt.get_size()[1]//2))
        # Paper Button
        pygame.draw.rect(win, PAPER_COLOR, PAPER_BUTTON)
        pygame.draw.rect(win, BUTTON_BORDER_COLOR, PAPER_BUTTON, BUTTON_BORDER_SIZE)
        txt = FONT.render("PAPER", True, BUTTON_FONT_COLOR)
        win.blit(txt, (BUTTONS_POSX + BUTTONS_GAP + (BUTTONS_WIDTH*3) // 2 - txt.get_size()[0]//2,
                       BUTTONS_POSY - BUTTONS_HEIGHT // 2 - txt.get_size()[1]//2))
        # Scissors Button
        pygame.draw.rect(win, SCISSOR_COLOR, SCISSORS_BUTTON)
        pygame.draw.rect(win, BUTTON_BORDER_COLOR, SCISSORS_BUTTON, BUTTON_BORDER_SIZE)
        txt = FONT.render("SCISSORS", True, BUTTON_FONT_COLOR)
        win.blit(txt, (BUTTONS_POSX + BUTTONS_GAP*2 + (BUTTONS_WIDTH*5) // 2 - txt.get_size()[0]//2,
                       BUTTONS_POSY - BUTTONS_HEIGHT // 2 - txt.get_size()[1]//2))
        # Scores
        color = WINNING_COLOR if self.sai < self.sp else LOSING_COLOR if self.sai > self.sp else FONT_COLOR
        txt = SCORESFONT.render(f"Player: {self.sp}", True, color)
        win.blit(txt, (PLAYER_SCORE_POSX - txt.get_size()[0], PLAYER_SCORE_POSY))
        color = WINNING_COLOR if self.sai > self.sp else LOSING_COLOR if self.sai < self.sp else FONT_COLOR
        txt = SCORESFONT.render(f"AI: {self.sai}", True, color)
        win.blit(txt, (AI_SCORE_POSX - txt.get_size()[0], AI_SCORE_POSY))

        # Played
        txt = MSGFONT.render(self.played, True, FONT_COLOR)
        win.blit(txt, (MSG_POSX-txt.get_size()[0]//2, MSG_POSY - txt.get_size()[1]//2))
        # Result
        txt = FONT.render(self.result, True, self.resultcolor)
        win.blit(txt, (MSG_POSX-txt.get_size()[0]//2, MSG_POSY - txt.get_size()[1]//2 + MSG_FONTSIZE*2))
        pygame.display.update()

def main():
    g = Game()
    g.draw()
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    g.handle_input(-1)
                elif event.key == pygame.K_p:
                    g.handle_input(0)
                elif event.key == pygame.K_s:
                    g.handle_input(1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                try:
                    g.handle_click(pos)
                except AttributeError:
                    continue
    pygame.quit()
    print(f"Thanks for playing! Final score:\nP - {g.sp} | AI - {g.sai}")
    return g.ai

model = main()
