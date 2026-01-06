
import pygame,random,os
pygame.font.init()
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Difficulty:
# Beginner - 9x9, 10 mines
# Intermediate - 16x16, 40 mines
# Expert - 16x30, 99 mines

# todo: change color of hint depending on number

WIDTH, HEIGHT = 800,800
GAPX,GAPY = 15,15
BG = (255,255,255)
CELL = (100,100,100)
SHOWNCELL = (150,150,150)
FLAGCOLOR = (255,0,0)
BORDER = True
BORDERWIDTH = 1
BORDERCOLOR = (0,0,0)

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


CLEAR = True # Clears all safe cells.

def mainloop(w: int, h: int, mines: int):
    global WIDTH, HEIGHT,BOARDW,BOARDH
    WIDTH -= (WIDTH-2*GAPX)%8
    HEIGHT -= (HEIGHT-2*GAPY)%8
    BOARDW,BOARDH = WIDTH-2*GAPX,HEIGHT-2*GAPY
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    board = Board(w,h,mines,window)
    first = True
    running = True
    board.draw(update=True)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                x,y = pygame.mouse.get_pos()
                x,y = (x-GAPX)//board.cellw, (y-GAPY)//board.cellh
                if not (0<=x<board.w) or not (0<=y<board.h) or board[(x,y)].flag or board[(x,y)].shown: continue
                if first:
                    board.setup(firstclick=(x,y))
                    first = False
                if CLEAR: board.clear(board[(x,y)])
                else: board[(x,y)].shown=True
                board.draw(update=True)
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[2] and not first:
                x,y = pygame.mouse.get_pos()
                x,y = (x-GAPX)//board.cellw, (y-GAPY)//board.cellh
                if not (0<=x<board.w) or not (0<=y<board.h) or board[(x,y)].shown: continue
                board[(x,y)].flag ^= True
                board.draw(update=True)
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
        self.cellw,self.cellh = BOARDW//self.w,BOARDH//self.h
        if win is not None:
            if not pygame.font.get_init(): pygame.font.init()
            self.cellfont = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", min(self.cellw,self.cellh))
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
    def draw(self, update: bool = False) -> None:
        """Renders board."""
        if self.win is None: return
        self.win.fill(BG)
        for x in range(self.w):
            for y in range(self.h):
                coord = (GAPX + x * self.cellw, GAPY + y * self.cellh)
                self.board[y][x].draw(self.win,coord,(self.cellw,self.cellh),self.cellfont)
        if update: pygame.display.update()


if __name__ == '__main__':
    mainloop(16,16,40)
