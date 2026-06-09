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
        """Size: radius | Opacity: 0-1 percentage | Types: 0 = pencil, 1 = ?"""
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
    
    def grayscale(self, update: bool = True):
        for y in range(self.h):
            for x in range(self.w):
                g = min(255,max(0,int(0.299*self.red[y][x]+0.587*self.green[y][x]+0.114*self.blue[y][x])))
                self.red[y][x],self.green[y][x],self.blue[y][x] = g,g,g
        self.draw(update=update)
    
    def invert(self, update: bool = True):
        for x in range(self.h):
            for y in range(self.w):
                self.red[y][x] = 255-self.red[y][x]
                self.green[y][x] = 255-self.green[y][x]
                self.blue[y][x] = 255-self.blue[y][x]
        self.draw(update=update)
    
    def edgedetect(self, update: bool = True):
        final = [[0 for _ in range(self.w)] for _ in range(self.h)]
        for x in range(self.h):
            for y in range(self.w):
                Y = 0
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if (i,j)==(0,0): X = 8
                        else: X = -1
                        try: Y += self.red[y+j][x+i]*X
                        except IndexError: Y += self.red[y][x]*X
                final[y][x] = int(min(255,max(0,Y)))
        self.red, self.green, self.blue = final,final,final
        self.draw(update=update)
    
    def sharpen(self, update: bool = True):
        r = [[0 for _ in range(self.w)] for _ in range(self.h)]
        g = [[0 for _ in range(self.w)] for _ in range(self.h)]
        b = [[0 for _ in range(self.w)] for _ in range(self.h)]
        for x in range(self.h):
            for y in range(self.w):
                R, G, B = 0,0,0
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if (i,j) in [(1,1),(-1,-1),(1,-1),(-1,1)]: X = 0
                        elif (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]: X = -1
                        else: X = 5
                        try:
                            R += self.red[y+j][x+i]*X
                            G += self.green[y+j][x+i]*X
                            B += self.blue[y+j][x+i]*X
                        except IndexError:
                            R += self.red[y][x]*X
                            G += self.green[y][x]*X
                            B += self.blue[y][x]*X
                R,G,B = int(min(255,max(0,R))),int(min(255,max(0,G))),int(min(255,max(0,B)))
                r[y][x],g[y][x],b[y][x] = R,G,B
        self.red,self.green,self.blue = r,g,b
        self.draw(update=update)
    
    def blur(self, update: bool = True):
        r = [[0 for _ in range(self.w)] for _ in range(self.h)]
        g = [[0 for _ in range(self.w)] for _ in range(self.h)]
        b = [[0 for _ in range(self.w)] for _ in range(self.h)]
        for x in range(self.h):
            for y in range(self.w):
                R, G, B = 0,0,0
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        X=1/9
                        try:
                            R += self.red[y+j][x+i]*X
                            G += self.green[y+j][x+i]*X
                            B += self.blue[y+j][x+i]*X
                        except IndexError:
                            R += self.red[y][x]*X
                            G += self.green[y][x]*X
                            B += self.blue[y][x]*X
                R,G,B = int(min(255,max(0,R))),int(min(255,max(0,G))),int(min(255,max(0,B)))
                r[y][x],g[y][x],b[y][x] = R,G,B
        self.red,self.green,self.blue = r,g,b
        self.draw(update=update)
    
    def gaussblur(self, update: bool = True):
        r = [[0 for _ in range(self.w)] for _ in range(self.h)]
        g = [[0 for _ in range(self.w)] for _ in range(self.h)]
        b = [[0 for _ in range(self.w)] for _ in range(self.h)]
        for x in range(self.h):
            for y in range(self.w):
                R, G, B = 0,0,0
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if (i,j) in [(1,1),(-1,-1),(1,-1),(-1,1)]: X = 1/16
                        elif (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]: X = 1/8
                        else: X = 1/4
                        try:
                            R += self.red[y+j][x+i]*X
                            G += self.green[y+j][x+i]*X
                            B += self.blue[y+j][x+i]*X
                        except IndexError:
                            R += self.red[y][x]*X
                            G += self.green[y][x]*X
                            B += self.blue[y][x]*X
                R,G,B = int(min(255,max(0,R))),int(min(255,max(0,G))),int(min(255,max(0,B)))
                r[y][x],g[y][x],b[y][x] = R,G,B
        self.red,self.green,self.blue = r,g,b
        self.draw(update=update)
    
    def sepia(self, update: bool = True):
        for y in range(self.h):
            for x in range(self.w):
                r = max(0,min(255,int(0.393*self.red[y][x]+0.769*self.green[y][x]+0.189*self.blue[y][x])))
                g = max(0,min(255,int(0.349*self.red[y][x]+0.686*self.green[y][x]+0.168*self.blue[y][x])))
                b = max(0,min(255,int(0.272*self.red[y][x]+0.534*self.green[y][x]+0.131*self.blue[y][x])))
                self.red[y][x],self.green[y][x],self.blue[y][x] = r,g,b
        self.draw(update=update)
    
    def colorswap(self, update: bool = True):
        for y in range(self.h):
            for x in range(self.w):
                self.red[y][x],self.green[y][x],self.blue[y][x] = self.blue[y][x],self.red[y][x],self.green[y][x]
        self.draw(update=update)

    def contrast(self, factor: float = 1.25, update: bool = True):
        """>1 higher contrast, <1 washed out, 0 = solid gray"""
        for y in range(self.h):
            for x in range(self.w):
                self.red[y][x] = max(0,min(255,128+factor*(self.red[y][x]-128)))
                self.green[y][x] = max(0,min(255,128+factor*(self.green[y][x]-128)))
                self.blue[y][x] = max(0,min(255,128+factor*(self.blue[y][x]-128)))
        self.draw(update=update)
    
    def saturate(self, factor: float = 1.25, update: bool = True):
        """>1 = saturate, <1 desaturate, 0 = grayscale, <0 hue inversion"""
        for y in range(self.h):
            for x in range(self.w):
                Y = 0.299*self.red[y][x]+0.587*self.green[y][x]+0.114*self.blue[y][x]
                self.red[y][x] = max(0,min(255,Y+factor*(self.red[y][x]-Y)))
                self.green[y][x] = max(0,min(255,Y+factor*(self.green[y][x]-Y)))
                self.blue[y][x] = max(0,min(255,Y+factor*(self.blue[y][x]-Y)))
        self.draw(update=update)

    def save(self, file: str = 'paint.txt') -> str:
        with open(file, 'w', encoding='utf-8') as f:
            f.write('.'.join([','.join(str(i) for i in row) for row in self.red])+'\n'+'.'.join([','.join(str(i) for i in row) for row in self.green])+'\n'+'.'.join([','.join(str(i) for i in row) for row in self.blue]))
        return file
    
    def load(self, file: str = 'paint.txt') -> str:
        with open(file, 'r', encoding='utf-8') as f:
            boards = f.readlines()
            self.red = [[int(i) for i in row.split(',')] for row in boards[0].split('.')]
            self.green = [[int(i) for i in row.split(',')] for row in boards[1].split('.')]
            self.blue = [[int(i) for i in row.split(',')] for row in boards[2].split('.')]
        return file

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
    modeselect = False
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
                elif event.key == pygame.K_f:
                    pos = pygame.mouse.get_pos()
                    try:
                        board.fill(pos, selected)
                    except (AttributeError,IndexError):
                        continue
                elif event.key == pygame.K_a:
                    try:
                        if INFO: print(f"Board loaded from '{board.load()}'.")
                        board.draw()
                    except Exception as e:
                        print(f"Unable to load: {e}")
                elif event.key == pygame.K_m:
                    modeselect = not modeselect
                    if INFO: print("Selecting transformation, 0-9." if modeselect else "Selecting colors, 0-9.")
                elif event.key == pygame.K_s:
                    try:
                        if INFO: print(f"Board saved to '{board.save()}'.")
                    except Exception as e:
                        print(f"Unable to save: {e}")
                elif event.key in [pygame.K_r,pygame.K_g,pygame.K_b]:
                    match event.key:
                        case pygame.K_r: selected = (255,0,0)
                        case pygame.K_g: selected = (0,255,0)
                        case _: selected = (0,0,255)
                    if INFO: print(f"Color selected: {selected}.")
                elif event.key in [pygame.K_0,pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9]:
                    match event.key:
                        case pygame.K_1:
                            if modeselect:
                                board.invert()
                                if INFO: print("Inverted.")
                            else:
                                if palette: colors[0] = selected
                                else: selected = colors[0]
                        case pygame.K_2:
                            if modeselect:
                                board.grayscale()
                                if INFO: print("Grayscaled.")
                            else:
                                if palette: colors[1] = selected
                                else: selected = colors[1]
                        case pygame.K_3:
                            if modeselect:
                                board.edgedetect()
                                if INFO: print("Detected edges.")
                            else:
                                if palette: colors[2] = selected
                                else: selected = colors[2]
                        case pygame.K_4:
                            if modeselect:
                                board.sharpen()
                                if INFO: print("Sharpened.")
                            else:
                                if palette: colors[3] = selected
                                else: selected = colors[3]
                        case pygame.K_5:
                            if modeselect:
                                board.blur()
                                if INFO: print("Blurred.")
                            else:
                                if palette: colors[4] = selected
                                else: selected = colors[4]
                        case pygame.K_6:
                            if modeselect:
                                board.gaussblur()
                                if INFO: print("Gaussian blurred.")
                            else:
                                if palette: colors[5] = selected
                                else: selected = colors[5]
                        case pygame.K_7:
                            if modeselect:
                                board.colorswap()
                                if INFO: print("Swapped color values.")
                            else:
                                if palette: colors[6] = selected
                                else: selected = colors[6]
                        case pygame.K_8:
                            if modeselect:
                                board.sepia()
                                if INFO: print("Sepia'd.")
                            else:
                                if palette: colors[7] = selected
                                else: selected = colors[7]
                        case pygame.K_9:
                            if modeselect:
                                board.contrast()
                                if INFO: print("Enhanced contrast.")
                            else:
                                if palette: colors[8] = selected
                                else: selected = colors[8]
                        case _:
                            if modeselect:
                                board.saturate()
                                if INFO: print("Saturated.")
                            else:
                                if palette: colors[9] = selected
                                else: selected = colors[9]
                    if INFO:
                        if modeselect:
                            pass
                        else:
                            if palette: print("Saved color, palette turned off.")
                            else: print(f"Selected color: {selected}")
                    if not modeselect and palette: palette = False
mainloop()
