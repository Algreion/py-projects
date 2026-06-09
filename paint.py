import pygame

win = None

HEIGHT = 800
WIDTH = 800
W = 64
H = 64
BORDER = 0
BORDERCOL = (200,200,200)
INFO = True

if W == H:
    HEIGHT -= HEIGHT%H
    WIDTH -= WIDTH%W
elif W > H:
    WIDTH -= WIDTH%W
    HEIGHT = H*(WIDTH//W)
else:
    HEIGHT -= HEIGHT%H
    WIDTH = W*(HEIGHT//H)

class Board:
    def __init__(self, colored: bool = False):
        self.w = W
        self.h = H
        self.W = WIDTH//W
        self.H = HEIGHT//H
        self.colored = colored
        self.makeboard()

    def makeboard(self):
        self.red = [[255 for _ in range(self.w)] for _ in range(self.h)]
        self.green = [[255 for _ in range(self.w)] for _ in range(self.h)]
        self.blue = [[255 for _ in range(self.w)] for _ in range(self.h)]
    
    def draw(self, update: bool = True):
        for h in range(self.h):
            for w in range(self.w):
                self.drawcell((w,h),update=False)
        if update: pygame.display.update()

    def drawcell(self, pos: tuple, update: bool = True):
        w,h = pos
        r,g,b = self.red[h][w],self.green[h][w],self.blue[h][w]
        pygame.draw.rect(win, (r,g,b), (self.W*w,self.H*h,self.W,self.H))
        if BORDER:
            pygame.draw.rect(win,BORDERCOL, (self.W*w,self.H*h,self.W,self.H),BORDER)
        if update: pygame.display.update()
    
    def color(self, cell: tuple, color: tuple, t: float = 1):
        w,h = cell
        r,g,b = color
        self.red[h][w] = int((1-t)*self.red[h][w]+t*r)
        self.green[h][w] = int((1-t)*self.green[h][w]+t*g)
        self.blue[h][w] = int((1-t)*self.blue[h][w]+t*b)

    def pick(self, pos: tuple) -> tuple:
        w,h = pos
        return (self.red[h][w],self.green[h][w],self.blue[h][w])
    
    def fill(self, pos: tuple, color: tuple, update: bool = True):
        x,y = pos[0]//self.W,pos[1]//self.H
        col = self.red[y][x],self.green[y][x],self.blue[y][x]
        def rec(a: int, b: int):
            cell = (a,b)
            if not 0<=a<self.w or not 0<=b<self.h:
                return
            if (self.red[b][a],self.green[b][a],self.blue[b][a]) != col: return
            self.color(cell, color)
            rec(a+1,b)
            rec(a-1,b)
            rec(a,b+1)
            rec(a,b-1)
        rec(x,y)
        self.draw()

    def paint(self, pos: tuple, color: tuple, brush: tuple, cell: tuple | None):
        """Size: radius | Opacity: 0-1 percentage | Types: 0 = pencil"""
        size, opacity, type = brush
        x,y = pos[0]//self.W,pos[1]//self.H
        if cell == (x,y): return cell
        if brush == (1,0,0):
            self.color((x,y), color)
            self.drawcell((x,y))
            return (x,y)
        match type:
            case 0:
                cells = [(x,y,color)]
                for h in range(self.h):
                    for w in range(self.w):
                        center = (self.W*w+self.W/2,self.H*h+self.H/2)
                        d = ((center[0]-pos[0])**2+(center[1]-pos[1])**2)**0.5
                        if d <= size: cells.append((w,h,color))
            case _:
                pass
        for c in cells:
            self.color((c[0],c[1]), c[2], t=opacity)
            self.drawcell((c[0],c[1]),update=False)
        pygame.display.update()
        return (x,y)
        

def mainloop():
    global win,WIDTH,HEIGHT
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    board = Board()
    board.draw()
    brush = (1, 1, 0) # size, transparency, type
    selected = (0,0,0)
    eraser = (255,255,255)
    eraserbrush = (30,1,0)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,128,0),(128,0,255),(255,255,255),(0,0,0)]
    cell = None
    palette = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONUP:
                cell = None
            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                try:
                    cell = board.paint(pos, selected, brush, cell)
                except (AttributeError,IndexError):
                    continue
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                try:
                    cell = board.paint(pos, eraser, eraserbrush, cell)
                except (AttributeError,IndexError):
                    continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
                    return
                elif event.key == pygame.K_c:
                    board.makeboard()
                    board.draw()
                    if INFO: print("Cleared board.")
                elif event.key == pygame.K_k:
                    brush = (brush[0],min(brush[1]+0.1,1),brush[2])
                    if INFO: print(f"Opacity: {brush[1]*100:.1f}%.")
                elif event.key == pygame.K_j:
                    brush = (brush[0],max(brush[1]-0.1,0),brush[2])
                    if INFO: print(f"Opacity: {brush[1]*100:.1f}%.")
                elif event.key == pygame.K_i:
                    brush = (max(brush[0]//2,1),brush[1],brush[2])
                    if INFO: print(f"Size: {brush[0]} px.")
                elif event.key == pygame.K_o:
                    brush = (min(brush[0]*2,max(WIDTH,HEIGHT)),brush[1],brush[2])
                    if INFO: print(f"Size: {brush[0]} px.")
                elif event.key == pygame.K_p:
                    pos = pygame.mouse.get_pos()
                    try:
                        x,y = pos[0]//board.W,pos[1]// board.H
                        selected = (board.red[y][x],board.green[y][x],board.blue[y][x])
                        if INFO: print(f"Picked color: {selected}.")
                    except (AttributeError,IndexError):
                        continue
                elif event.key == pygame.K_q:
                    palette = not palette
                    if INFO:
                        if palette: print(f"Activated palette. Type a digit 0-9 to save color {selected} to the slot.")
                        else: print("Palette turned off.")
                elif event.key == pygame.K_x:
                    for x in range(board.h):
                        for y in range(board.w):
                            board.red[y][x] = 255-board.red[y][x]
                            board.green[y][x] = 255-board.green[y][x]
                            board.blue[y][x] = 255-board.blue[y][x]
                    board.draw()
                    if INFO: print("Inverted board.")
                elif event.key == pygame.K_f:
                    pos = pygame.mouse.get_pos()
                    try:
                        board.fill(pos, selected)
                    except (AttributeError,IndexError):
                        continue
                elif event.key in [pygame.K_0,pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9]:
                    match event.key:
                        case pygame.K_1:
                            if palette: colors[0] = selected
                            else: selected = colors[0]
                        case pygame.K_2:
                            if palette: colors[1] = selected
                            else: selected = colors[1]
                        case pygame.K_3:
                            if palette: colors[2] = selected
                            else: selected = colors[2]
                        case pygame.K_4:
                            if palette: colors[3] = selected
                            else: selected = colors[3]
                        case pygame.K_5:
                            if palette: colors[4] = selected
                            else: selected = colors[4]
                        case pygame.K_6:
                            if palette: colors[5] = selected
                            else: selected = colors[5]
                        case pygame.K_7:
                            if palette: colors[6] = selected
                            else: selected = colors[6]
                        case pygame.K_8:
                            if palette: colors[7] = selected
                            else: selected = colors[7]
                        case pygame.K_9:
                            if palette: colors[8] = selected
                            else: selected = colors[8]
                        case _:
                            if palette: colors[9] = selected
                            else: selected = colors[9]
                    if INFO:
                        if palette: print("Saved color, palette turned off.")
                        else: print(f"Selected color: {selected}")
                    if palette: palette = False
mainloop()
