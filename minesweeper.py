
import os,random,time
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
pygame.font.init()

CUSTOM = (16,16,225) # (Width, Height, Mines)

#TODO: Sounds

WIDTH, HEIGHT = 800,800
GAPX,GAPY = 25,25
BG = (255,255,255)
CELL = (150,150,150)
SHOWNCELL = (200,200,200)
FLAGCOLOR = (255,0,0)
BORDER = True
BORDERWIDTH = 1
BORDERCOLOR = (0,0,0)
TIMERCOLOR = (0,0,0)
WINFONT = (255,210,0)
LOSEFONT = (64, 0, 0)
ENDFONT = (0,0,0)
ENDBUTTONS = (255,255,255)
MENUBG = (255,255,255)
MENUFONT = (0,0,0)
BOARDW,BOARDH = WIDTH-2*GAPX,HEIGHT-2*GAPY

SYMBOLS = {'flag':'🚩','mine':'💣'}
HINTCOLOR = {
    -1: (0,0,0), # Bomb
    0: SHOWNCELL,
    1: (0, 0, 255),
    2: (0, 128, 0),
    3: (255, 0, 0),
    4: (0, 0, 128),
    5: (128, 0, 0),
    6: (0, 128, 128),
    7: (123,110,0),
    8: (200,0,200),
}
DIFFICULTY = {
    1: (9,9,10),
    2: (16,16,40),
    3: (25,25,125),
    4: CUSTOM
}

CLEAR = True # Clears all safe cells.
INFO = (True, True) # (Flag count, timer)
LOSETIME = 0.5 # Time before bomb explodes

def mainloop(w: int, h: int, mines: int, cont = None) -> int:
    """Handles a single round. Res: -1 = quit, Board = play again, 1 = menu."""
    global WIDTH, HEIGHT,BOARDW,BOARDH
    WIDTH -= (WIDTH-2*GAPX)%w
    HEIGHT -= (HEIGHT-2*GAPY)%h
    BOARDW,BOARDH = WIDTH-2*GAPX,HEIGHT-2*GAPY
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Minesweeper")
    first = True
    start = 0
    now = 0
    running = True
    if cont is None: board = Board(w,h,mines,window)
    else:
        board = cont
        first = False
        if INFO[1]: start = time.monotonic()-board.seconds
    board.draw(update=True, sec=now)
    res = -1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE,pygame.K_x]:
                    if first:
                        res = 1
                    else:
                        board.seconds = now
                        res = board
                    running = False
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                x,y = pygame.mouse.get_pos()
                x,y = (x-GAPX)//board.cellw, (y-GAPY)//board.cellh
                if not (0<=x<board.w) or not (0<=y<board.h) or board[(x,y)].flag or board[(x,y)].shown: continue
                if first:
                    board.setup(firstclick=(x,y))
                    if INFO[1]: start = time.monotonic()
                    first = False
                if CLEAR: board.clear(board[(x,y)])
                else: board[(x,y)].shown=True
                board.draw(update=True, sec=now)
                if board[(x,y)].hint == -1:
                    time.sleep(LOSETIME)
                    losing = True
                    buttons = board.drawwin(sec=now,loss=True)
                    rev = False
                    while losing:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                losing,running = False, False
                                break
                            elif rev and event.type in [pygame.MOUSEBUTTONDOWN,pygame.KEYDOWN]:
                                buttons = board.drawwin(sec=now,loss=True)
                                rev = False
                            elif not rev and event.type == pygame.MOUSEBUTTONDOWN:
                                mx,my = pygame.mouse.get_pos()
                                if buttons[0].collidepoint(mx,my): # Play Again
                                    losing = False
                                    del board
                                    board = Board(w,h,mines,window)
                                    first,start,now,res = True,0,0,-1
                                    board.draw(update=True, sec=now)
                                    break
                                elif buttons[1].collidepoint(mx,my): # Menu
                                    res = 1
                                    losing,running = False, False
                                    break
                                elif buttons[2].collidepoint(mx,my): # Quit
                                    losing,running = False, False
                                    break
                                elif buttons[3].collidepoint(mx,my): # Reveal
                                    rev = True
                                    for y in range(board.h):
                                        for x in range(board.w):
                                            c = board[(x,y)]
                                            if c.flag and c.hint==-1: continue
                                            c.flag,c.shown = False,True
                                    board.draw(update=True,sec=now)
                elif board.checkwin():
                    winning = True
                    buttons = board.drawwin(sec=now)
                    while winning:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                winning,running = False, False
                                break
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                mx,my = pygame.mouse.get_pos()
                                if buttons[0].collidepoint(mx,my): # Play Again
                                    winning = False
                                    del board
                                    board = Board(w,h,mines,window)
                                    first,start,now,res = True,0,0,-1
                                    board.draw(update=True, sec=now)
                                    break
                                elif buttons[1].collidepoint(mx,my): # Menu
                                    res = 1
                                    winning,running = False, False
                                    break
                                elif buttons[2].collidepoint(mx,my): # Quit
                                    winning,running = False, False
                                    break
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[2] and not first:
                x,y = pygame.mouse.get_pos()
                x,y = (x-GAPX)//board.cellw, (y-GAPY)//board.cellh
                if not (0<=x<board.w) or not (0<=y<board.h) or board[(x,y)].shown: continue
                board[(x,y)].flag ^= True
                if board[(x,y)].flag: board.flags -= 1
                else: board.flags += 1
                board.draw(update=True, sec=now)
        if not running:
            break
        if start != 0 and INFO[1] and int(time.monotonic()-start) != now:
            now = int(time.monotonic()-start)
            board.draw(update=True,sec=now)
    return res

def drawmenu(win, endfont, cont: bool = False) -> list:
    """Renders start menu. Returns buttons list."""
    buttons = []
    win.fill(MENUBG)
    i = endfont.render("Minesweeper", True, MENUFONT)
    surf = i.get_rect(center=(WIDTH//2,3*HEIGHT//10))
    win.blit(i,surf)
    buttonwidth = WIDTH//2
    i = endfont.render("EASY", True, ENDFONT)
    surf = i.get_rect(center=(WIDTH//2,5*HEIGHT//10))
    text = surf.copy()
    center, surf.width = surf.center, buttonwidth
    surf.center = center
    pygame.draw.rect(win, ENDBUTTONS, surf, 0)
    pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
    win.blit(i, text)
    buttons.append(surf)
    i = endfont.render("INTERMEDIATE", True, ENDFONT)
    surf = i.get_rect(center=(WIDTH//2,6*HEIGHT//10))
    text = surf.copy()
    center, surf.width = surf.center, buttonwidth
    surf.center = center
    pygame.draw.rect(win, ENDBUTTONS, surf, 0)
    pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
    win.blit(i, text)
    buttons.append(surf)
    i = endfont.render("EXPERT", True, ENDFONT)
    surf = i.get_rect(center=(WIDTH//2,7*HEIGHT//10))
    text = surf.copy()
    center, surf.width = surf.center, buttonwidth
    surf.center = center
    pygame.draw.rect(win, ENDBUTTONS, surf, 0)
    pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
    win.blit(i, text)
    buttons.append(surf)
    i = endfont.render("CUSTOM", True, ENDFONT)
    surf = i.get_rect(center=(WIDTH//2,8*HEIGHT//10))
    text = surf.copy()
    center, surf.width = surf.center, buttonwidth
    surf.center = center
    pygame.draw.rect(win, ENDBUTTONS, surf, 0)
    pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
    win.blit(i, text)
    buttons.append(surf)
    i = endfont.render("QUIT", True, ENDFONT)
    surf = i.get_rect(center=(WIDTH//2,9*HEIGHT//10))
    text = surf.copy()
    center, surf.width = surf.center, buttonwidth
    surf.center = center
    pygame.draw.rect(win, ENDBUTTONS, surf, 0)
    pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
    win.blit(i, text)
    buttons.append(surf)
    if cont:
        i = endfont.render("CONTINUE", True, ENDFONT)
        surf = i.get_rect(center=(WIDTH//2,4*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(win, ENDBUTTONS, surf, 0)
        pygame.draw.rect(win, (255-ENDBUTTONS[0],255-ENDBUTTONS[1],255-ENDBUTTONS[2]), surf, 2)
        win.blit(i, text)
        buttons.append(surf)
    pygame.display.update()
    return buttons

def loop():
    """Full game loop."""
    running = True
    if not pygame.font.get_init(): pygame.font.init()
    endfont = pygame.font.SysFont("verdana",min(WIDTH,HEIGHT)//15)
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    buttons = drawmenu(window,endfont)
    cont = False
    res = -2
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE,pygame.K_x]:
                    pygame.quit()
                    running = False
                    break
                elif event.key == pygame.K_1: # Easy
                    res = mainloop(*DIFFICULTY[1])
                elif event.key == pygame.K_2: # Intermediate
                    res = mainloop(*DIFFICULTY[2])
                elif event.key == pygame.K_3: # Expert
                    res = mainloop(*DIFFICULTY[3])
                elif event.key == pygame.K_4: # Custom, edit in file
                    res = mainloop(*DIFFICULTY[4])
                elif event.key == pygame.K_5: # Quit
                    pygame.quit()
                    running = False
                    break
                elif cont and event.key == pygame.K_0: # Continue
                    res = mainloop(res.w,res.h,res.mines,res)
                if res == -1:
                    running = False
                    pygame.quit()
                    break
                elif res == 1:
                    cont = False
                    res = -2
                    buttons = drawmenu(window,endfont)
                elif isinstance(res,Board):
                    cont = True
                    buttons = drawmenu(window,endfont,cont)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = pygame.mouse.get_pos()
                if buttons[0].collidepoint(mx,my): # Easy
                    res = mainloop(*DIFFICULTY[1])
                elif buttons[1].collidepoint(mx,my): # Intermediate
                    res = mainloop(*DIFFICULTY[2])
                elif buttons[2].collidepoint(mx,my): # Expert
                    res = mainloop(*DIFFICULTY[3])
                elif buttons[3].collidepoint(mx,my): # Custom
                    res = mainloop(*DIFFICULTY[4])
                elif buttons[4].collidepoint(mx,my): # Quit
                    pygame.quit()
                    running = False
                    break
                elif cont and buttons[5].collidepoint(mx,my): # Continue
                    res = mainloop(res.w,res.h,res.mines,res)
                if res == -1:
                    running = False
                    pygame.quit()
                    break
                elif res == 1:
                    cont = False
                    res = -2
                    buttons = drawmenu(window,endfont)
                elif isinstance(res,Board):
                    cont = True
                    buttons = drawmenu(window,endfont,cont)

class Cell:
    def __init__(self, index: tuple, mine: bool = False, shown: bool = False):
        self.shown = shown
        self.hint = -1 if mine else 0
        self.index = index
        self.flag = False
    def __repr__(self):
        return str(self.hint) if self.hint != -1 else 'X'
    def check(self, board: list) -> None:
        """Checks adjacent cells and edits self.hint"""
        self.hint = sum([c.hint == -1 for c in self.adj(board)])
    def adj(self, board: list) -> list:
        """Returns list of adjacent cells."""
        res = []
        for dx,dy in [(0,1),(0,-1),(1,-1),(1,0),(1,1),(-1,-1),(-1,0),(-1,1)]:
            x,y = self.index[0]+dx,self.index[1]+dy
            if 0<=x<len(board[0]) and 0<=y<len(board): res.append(board[y][x])
        return res
    def draw(self, win, coord: tuple, size: tuple, font):
        """Renders single cell. Size is (w,h) of board."""
        color = SHOWNCELL if self.shown else CELL
        pygame.draw.rect(win, color, (coord[0], coord[1], size[0], size[1]))
        if BORDER: pygame.draw.rect(win, BORDERCOLOR, (coord[0], coord[1], size[0], size[1]), BORDERWIDTH)
        if self.shown:
            txt = SYMBOLS['mine'] if self.hint == -1 else str(self.hint)
            p = font.render(txt, True, HINTCOLOR[self.hint])
            center = p.get_rect(center=(coord[0]+size[0]//2,coord[1]+size[1]//2))
            win.blit(p,center)
        elif self.flag:
            p = font.render(SYMBOLS['flag'], True, FLAGCOLOR)
            center = p.get_rect(center=(coord[0]+size[0]//2,coord[1]+size[1]//2))
            win.blit(p,center)
class Board:
    def __init__(self, w: int, h: int, mines: int = 0, win = None):
        self.w = w
        self.h = h
        self.mines = mines
        self.win = win
        self.board = [[Cell((x,y)) for x in range(self.w)] for y in range(self.h)]
        self.cellw = min(BOARDW//self.w,BOARDH//self.h)
        self.cellh = self.cellw
        self.flags = self.mines
        self.seconds = 0
        if win is not None:
            if not pygame.font.get_init(): pygame.font.init()
            self.cellfont = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", min(self.cellw,self.cellh))
            self.infofont = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", min(GAPX,GAPY)-1)
            self.endfont = pygame.font.SysFont("verdana",min(WIDTH,HEIGHT)//15)
    def __getitem__(self, index: tuple[int]):
        return self.board[index[1]][index[0]]
    def __repr__(self):
        return f"Board(size={self.w}x{self.h},mines={self.mines})"
    def __str__(self):
        return '\n'.join([''.join([str(c) for c in self.board[y]]) for y in range(self.h)])
    def setup(self, firstclick: tuple | None = None, mines: int | None = None) -> None:
        """Sets up the mines and hints. """
        if firstclick is None: firstclick = (0,0)
        if mines is None: mines = self.mines
        mines = min(mines,self.w*self.h-1)
        self.flags = mines
        last = []
        for dx,dy in [(0,1),(0,-1),(1,-1),(1,0),(1,1),(-1,-1),(-1,0),(-1,1)]:
            x,y = firstclick[0]+dx,firstclick[1]+dy
            last.append((x,y))
        choices = [self.board[h][w] for w in range(self.w) for h in range(self.h) if (w,h) not in [firstclick]+last]
        last = [self.board[h][w] for (w,h) in last]
        random.shuffle(choices)
        for _ in range(mines):
            if choices: choices.pop().hint = -1
            else: last.pop().hint = -1
        for c in choices+last+[self.board[firstclick[1]][firstclick[0]]]:
            c.check(self.board)
    def clear(self, cell: Cell) -> None:
        """Shows all spots surrounding safe cells."""
        done = set()
        def clear0(c: Cell) -> None:
            nonlocal done
            if c in done: return
            done.add(c)
            c.shown = True
            if c.hint == 0:
                for ce in c.adj(self.board): clear0(ce)
        clear0(cell)
    def checkwin(self) -> bool:
        """Checks if all non-bombs were shown."""
        for y in range(self.h):
            for x in range(self.w):
                c = self[(x,y)]
                if c.hint != -1 and not c.shown: return False
        return True
    def draw(self, update: bool = False, sec: int = 0) -> None:
        """Renders board."""
        if self.win is None: return
        self.win.fill(BG)
        for x in range(self.w):
            for y in range(self.h):
                coord = (GAPX + x * self.cellw, GAPY + y * self.cellh)
                self.board[y][x].draw(self.win,coord,(self.cellw,self.cellh),self.cellfont)
        if INFO[0]:
            p = self.infofont.render(f"{SYMBOLS['flag']}: {self.flags}", True, FLAGCOLOR)
            center = p.get_rect(center=(0,GAPY//2))
            center.left = GAPX
            self.win.blit(p,center)
        if INFO[1]:
            m,s = sec//60,sec%60
            p = self.infofont.render(f"{m:02d}:{s:02d}", True, TIMERCOLOR)
            center = p.get_rect(center=(WIDTH//2,GAPY//2))
            self.win.blit(p,center)
        if update: pygame.display.update()
    def drawwin(self, sec: int = 0, loss: bool = False) -> list:
        """Draw win/lose screen. Returns buttons list."""
        buttons = []
        self.draw(sec=sec)
        w, h = self.win.get_size()
        small = pygame.transform.smoothscale(self.win, (w//3, h//3))
        blurred = pygame.transform.smoothscale(small, (w, h))
        self.win.blit(blurred, (0, 0))
        if loss: i = self.endfont.render("BOOM!", True, LOSEFONT)
        else: i = self.endfont.render("YOU WON!", True, WINFONT)
        surf = i.get_rect(center=(WIDTH//2,3*HEIGHT//10))
        self.win.blit(i,surf)
        if INFO[1] and not loss:
            m,s = sec//60,sec%60
            i = self.endfont.render(f"Time taken: {f"{m:02d}:{s:02d}"}", True, WINFONT)
            surf = i.get_rect(center=(WIDTH//2,4*HEIGHT//10))
            self.win.blit(i,surf)
        buttonwidth = WIDTH//2
        i = self.endfont.render("PLAY AGAIN", True, ENDFONT)
        surf = i.get_rect(center=(WIDTH//2,5*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, ENDBUTTONS, surf, 0)
        self.win.blit(i, text)
        buttons.append(surf)
        i = self.endfont.render("MENU", True, ENDFONT)
        surf = i.get_rect(center=(WIDTH//2,6*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, ENDBUTTONS, surf, 0)
        self.win.blit(i, text)
        buttons.append(surf)
        i = self.endfont.render("QUIT", True, ENDFONT)
        surf = i.get_rect(center=(WIDTH//2,7*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, ENDBUTTONS, surf, 0)
        self.win.blit(i, text)
        buttons.append(surf)
        if loss:
            i = self.endfont.render("REVEAL", True, ENDFONT)
            surf = i.get_rect(center=(WIDTH//2,4*HEIGHT//10))
            text = surf.copy()
            center, surf.width = surf.center, buttonwidth
            surf.center = center
            pygame.draw.rect(self.win, ENDBUTTONS, surf, 0)
            self.win.blit(i, text)
            buttons.append(surf)
        pygame.display.update()
        return buttons

if __name__ == '__main__':
    loop()
