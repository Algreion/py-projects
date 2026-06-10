
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame,random
from collections import deque
from copy import deepcopy

win = None

LOAD = '' # Will load image file if specified, eg. 'image.png'

SAVEPATH = 'image.png' # Name for a saved image file.

HEIGHT = 800
WIDTH = 800
W = 64
H = 64
BORDER = 0 # >0 for cell borders
BORDERCOL = (200,200,200) # border color
INFO = True
MAXUNDO = 10 # How many previous board states are saved

RED=GREEN=BLUE=None

def loadimage(file: str) -> bool:
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed, can't load image.")
        return False
    global HEIGHT,WIDTH,W,H,RED,GREEN,BLUE,LOAD
    maxwidth = 1920
    maxheight = 1080
    try:
        img = Image.open(file).convert("RGB")
    except (FileNotFoundError, OSError):
        print(f'{file} not found.')
        LOAD = ''
        return False
    px = list(img.getdata())
    width, height = img.size
    if width > maxwidth or height > maxheight:
        print(width,height)
        print(f"Image too large, max size is {(maxwidth, maxheight)}.")
        LOAD = ''
        return False
    RED = [[px[y * width + x][0] for x in range(width)] for y in range(height)]
    GREEN = [[px[y * width + x][1] for x in range(width)] for y in range(height)]
    BLUE = [[px[y * width + x][2] for x in range(width)] for y in range(height)]
    W = width
    H = height
    scale = 1 # Upscaling
    while (scale+1)*width <= maxwidth and (scale+1)*height <= maxheight:
        scale += 1
    WIDTH, HEIGHT = width*scale, height*scale
    return True

if LOAD and loadimage(LOAD):
    print(f"Image '{LOAD}' loaded successfully.")

if max(W,H)>256: MAXUNDO = min(MAXUNDO,3) # Avoids overhead

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

    def makeboard(self, cleared: bool = False):
        if cleared or not LOAD:
            self.red = [[255 for _ in range(self.w)] for _ in range(self.h)]
            self.green = [[255 for _ in range(self.w)] for _ in range(self.h)]
            self.blue = [[255 for _ in range(self.w)] for _ in range(self.h)]
        else:
            self.red = RED
            self.blue = BLUE
            self.green = GREEN
    
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
        if col == color: return
        stack = [(x,y)]
        while stack:
            a,b = stack.pop()
            if not 0<=a<self.w or not 0<=b<self.h: continue
            if (self.red[b][a],self.green[b][a],self.blue[b][a]) != col: continue
            self.color((a, b), color)
            stack.extend([(a+1, b), (a-1, b), (a, b+1), (a, b-1)])
        self.draw(update=update)

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
        for y in range(self.h):
            for x in range(self.w):
                self.red[y][x] = 255-self.red[y][x]
                self.green[y][x] = 255-self.green[y][x]
                self.blue[y][x] = 255-self.blue[y][x]
        self.draw(update=update)
    
    def edgedetect(self, update: bool = True):
        final = [[0 for _ in range(self.w)] for _ in range(self.h)]
        for y in range(self.h):
            for x in range(self.w):
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
        for y in range(self.h):
            for x in range(self.w):
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
        X = 1/9
        for y in range(self.h):
            for x in range(self.w):
                R, G, B = 0,0,0
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
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
        for y in range(self.h):
            for x in range(self.w):
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

    def noise(self, update: bool = True):
        for y in range(self.h):
            for x in range(self.w):
                self.red[y][x],self.green[y][x],self.blue[y][x] = random.randint(1,255),random.randint(1,255),random.randint(1,255)
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
        """Saves as txt file."""
        with open(file, 'w', encoding='utf-8') as f:
            f.write('.'.join([','.join(str(i) for i in row) for row in self.red])+'\n'+'.'.join([','.join(str(i) for i in row) for row in self.green])+'\n'+'.'.join([','.join(str(i) for i in row) for row in self.blue]))
        return file
    
    def load(self, file: str = 'paint.txt') -> str:
        with open(file, 'r', encoding='utf-8') as f:
            boards = f.readlines()
            red = [[int(i) for i in row.split(',')] for row in boards[0].split('.')]
            if len(red) != self.h or len(red[0]) != self.w: return (len(red[0]),len(red))
            green = [[int(i) for i in row.split(',')] for row in boards[1].split('.')]
            blue = [[int(i) for i in row.split(',')] for row in boards[2].split('.')]
            self.red,self.green,self.blue = red,green,blue
        return file
    
    def saveimage(self, file: str = 'image.png'):
        """Saves as png file using PIL."""
        try:
            from PIL import Image
        except ImportError:
            print("PIL not installed, can't save image.")
            return
        height, width = len(self.red), len(self.red[0])
        img = Image.new('RGB', (width,height))
        px = [(self.red[y][x], self.green[y][x], self.blue[y][x]) for y in range(height) for x in range(width)]
        img.putdata(px)
        img.save(file)
        return True

def mainloop():
    global win,WIDTH,HEIGHT,INFO
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
    undo = deque(maxlen=MAXUNDO)
    redo = deque(maxlen=MAXUNDO)
    drawing = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return
            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if 0 <= pos[0] < WIDTH and 0 <= pos[1] < HEIGHT:
                    if not drawing:
                        undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                        redo.clear()
                        drawing = True
                    cell = board.paint(pos, selected, brush, cell)
            elif event.type == pygame.MOUSEBUTTONUP:
                cell = None
                drawing = False
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                if 0 <= pos[0] < WIDTH and 0 <= pos[1] < HEIGHT:
                    if not drawing:
                        undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                        redo.clear()
                        drawing = True
                    cell = board.paint(pos, eraser, eraserbrush, cell)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
                    return
                elif event.key == pygame.K_h:
                    print(f"""Keybinds: 0-9 = Select color | C = Clear board | K/L = Opacity | Z/X = Brush size
P = Color picker | F = Fill tool | Q = Save color to slot | M + 0-9 = Select filter | U/I: Undo/Redo
S = Save | A = Load | D = Toggle Info logs | R/G/B = Red/Green/Blue | Tab: Select random color
W = Save as {SAVEPATH} | \\: Scramble board | ESC = Quit""")
                elif event.key == pygame.K_c:
                    undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                    redo.clear()
                    board.makeboard(cleared=True)
                    board.draw()
                    if INFO: print("Cleared board.")
                elif event.key == pygame.K_d:
                    INFO = not INFO
                    print(f"Info logs turned {'ON' if INFO else 'OFF'}.")
                elif event.key == pygame.K_l:
                    brush = (brush[0],min(brush[1]+0.1,1),brush[2])
                    if INFO: print(f"Opacity: {brush[1]*100:.1f}%.")
                elif event.key == pygame.K_k:
                    brush = (brush[0],max(brush[1]-0.1,0),brush[2])
                    if INFO: print(f"Opacity: {brush[1]*100:.1f}%.")
                elif event.key == pygame.K_z:
                    brush = (max(brush[0]//2,1),brush[1],brush[2])
                    if INFO: print(f"Size: {brush[0]} px.")
                elif event.key == pygame.K_x:
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
                elif event.key == pygame.K_TAB:
                    selected = (random.randint(1,255),random.randint(1,255),random.randint(1,255))
                    if INFO: print(f"Selected random color: {selected}.")
                elif event.key == pygame.K_BACKSLASH:
                    undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                    redo.clear()
                    board.noise()
                    if INFO: print("Scrambled board.")
                elif event.key == pygame.K_f:
                    undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                    redo.clear()
                    pos = pygame.mouse.get_pos()
                    try:
                        board.fill(pos, selected)
                    except (AttributeError,IndexError):
                        continue
                elif event.key == pygame.K_u and undo:
                    redo.append((deepcopy(board.red), deepcopy(board.green), deepcopy(board.blue)))
                    board.red,board.green,board.blue = undo.pop()
                    board.draw()
                elif event.key == pygame.K_i and redo:
                    undo.append((deepcopy(board.red), deepcopy(board.green), deepcopy(board.blue)))
                    board.red, board.green, board.blue = redo.pop()
                    board.draw()
                elif event.key == pygame.K_a:
                    undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                    redo.clear()
                    try:
                        board.draw()
                        file = board.load()
                        if isinstance(file,tuple): print(f"Size mismatch when loading! Board is {(board.w,board.h)}, file is {file}.")
                        else:
                            if INFO: print(f"Board loaded from '{file}'.")
                            board.draw()
                    except Exception as e:
                        print(f"Unable to load: {e}")
                elif event.key == pygame.K_s:
                    try:
                        print(f"Board saved to '{board.save()}'.")
                    except Exception as e:
                        print(f"Unable to save: {e}")
                elif event.key == pygame.K_w:
                    if board.saveimage(file=SAVEPATH):
                        print(f"Image saved to '{SAVEPATH}' successfully.")
                elif event.key == pygame.K_m:
                    modeselect = not modeselect
                    if INFO: print("Selecting transformation, 0-9." if modeselect else "Selecting colors, 0-9.")
                elif event.key in [pygame.K_r,pygame.K_g,pygame.K_b]:
                    match event.key:
                        case pygame.K_r: selected = (255,0,0)
                        case pygame.K_g: selected = (0,255,0)
                        case _: selected = (0,0,255)
                    if INFO: print(f"Color selected: {selected}.")
                elif event.key in [pygame.K_0,pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9]:
                    if modeselect:
                        undo.append((deepcopy(board.red),deepcopy(board.green),deepcopy(board.blue)))
                        redo.clear()
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

print("Paint v1.0 - Press H for keybinds.")
mainloop()
