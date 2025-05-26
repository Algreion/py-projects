import pygame

WIDTH, HEIGHT = 800, 800
GRIDX, GRIDY = 10, 10
STARTPOS = (0, HEIGHT)
STARTANGLE = 30
GRAVITY = 9.81
BACKGROUND = (255,255,255)
GRIDCOLOR = (180,180,180)

WIN = None
if WIDTH % GRIDX: WIDTH = WIDTH + (GRIDX-WIDTH%GRIDX)
if HEIGHT % GRIDY: HEIGHT + (GRIDY-HEIGHT%GRIDY)
XX, YY = WIDTH//GRIDX, HEIGHT//GRIDY

class Simulation:
    def __init__(self, win: pygame.display):
        self.win = win
        self.gravity = GRAVITY
    
    def render(self):
        print(WIDTH, HEIGHT)
        self.win.fill(BACKGROUND)
        for i in range(GRIDX):
            pygame.draw.line(self.win, GRIDCOLOR, (XX*i, 0), (XX*i, HEIGHT))
        for j in range(GRIDY):
            pygame.draw.line(self.win, GRIDCOLOR, (0, YY*j), (WIDTH, YY*j))
        pygame.display.update()

class Projectile:
    def __init__(self, sim):
        self.sim = sim
        self.pos = STARTPOS
        self.angle = 30
        self.speed = 0

def mainloop():
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    sim = Simulation(WIN)
    proj = Projectile(sim)
    run = True
    sim.render()
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                break

mainloop()
