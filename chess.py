
from rich import print
import os
#TODO: mainloop, check if everything works, more pieces, castle, promotion, en passant, check & checkmate (forced moves), text colors, improved move/capture option render, make in pygame?

RICH = True # Colors
BLACK = 'cyan'
WHITE = 'cornsilk1'
EMPTY = 'bold'

class Board:
    def __init__(self, N: tuple = (8,8), filled: bool = False, type: str | None = 'Chess'):
        """Chessboard."""
        self.W, self.H = N
        self.type = type
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
        self.symbol_dict = {"king":"♔","queen":"♕","rook":"♖","bishop":"♗","knight":"♘","pawn":"♙"}
        self.simplified_dict = {"king":"k","queen":"q","rook":"r","bishop":"b","knight":"n","pawn":"p",None: '.'}
        self.gap = 2 # board str
        self.blank = '▢'
        self.white_king = 'e1'
        self.black_king = 'e8'
        self.white_captured,self.black_captured = set(),set()
        self.move_dict = {
        "king":    [(1,0), (-1,0), (0,1), (0,-1),
                    (1,1), (1,-1), (-1,1), (-1,-1)],
        "knight":  [(2,1), (2,-1), (-2,1), (-2,-1),
                    (1,2), (1,-2), (-1,2), (-1,-2)],
        "bishop": [(1,1),(1,-1),(-1,1),(-1,-1)],
        "rook": [(1,0),(0,1),(-1,0),(0,-1)],
        "queen": [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        }
        self.points = {'pawn':1,'knight':3,'bishop':3,'rook':5,'queen':9}
        if filled: self.setup()
    def __repr__(self):
        return f"Board(width={self.W},height={self.H},type={self.type})"
    def __str__(self):
        res = []
        for i,line in enumerate(self.board[::-1]):
            res.append("|")
            res[i] += ''.join([str(p).center(self.gap) if p is not None else self.blank.center(self.gap) for p in line])+"| "+str(self.H-i)
        return "\n".join(res)+'\n '+''.join([c.center(self.gap) for c in "abcdefgh"])
    def __getitem__(self, notation: str | tuple):
        return self.lookup(notation)
    def __setitem__(self, notation: str | tuple, item):
        x,y = self.ind(notation)
        self.board[y][x] = item
    def __delitem__(self, notation: str | tuple):
        x,y = self.ind(notation)
        self.board[y][x] = None
    def check(self, square: tuple | str, col: int | None = None, pieces: bool = False) -> bool | list:
        """"Returns whether the square is seen by enemy pieces. Pieces toggle returns a list of attacking pieces."""
        square = self.ind(square)
        piece = self.lookup(square)
        attacking = []
        if piece is None and col is None: return False
        if col is not None: color = col
        else: color = piece.color
        for w in range(self.W):
            for h in range(self.H):
                if self.lookup((w,h)) is not None and self.lookup((w,h)).color != color:
                    if piece is None or piece.color == color: T = False
                    else: T = True
                    if self.ind(square) in self.capture_options((w,h),T)+self.move_options((w,h)):
                        if pieces: attacking.append((w,h))
                        else: return True
        return attacking if pieces else False
    def checkmate(self, color: int = 1) -> bool:
        """Flag of whether the king is in checkmate."""
        if color == 1: king = self.white_king
        else: king = self.black_king
        attacking = self.check(king,pieces=True)
        if not attacking: return False
        for o in self.capture_options(king)+self.move_options(king):
            if not self.check(o,color): return False
        if len(attacking) == 1:
            xp,yp = attacking[0]
            xk,yk = self.ind(king)
            if self.check((xp,yp)): return False
            if self.lookup((xp,yp)).type in ['rook','queen','bishop']:
                dirs = self.directions((xp,yp))
                opts = []
                for direction in dirs:
                    if (xk,yk) in direction:
                        opts = direction
                        break
                for w in range(self.W):
                    for h in range(self.H):
                        if (P:=self.lookup[(w,h)]) is not None and P.color == color:
                            if any([move in opts for move in self.move_options((w,h))]): return False
        return True
    def directions(self, square: str | tuple) -> list:
        """Returns a list of all line directions for sliding pieces."""
        x,y = self.ind(square)
        d = self.move_dict[self.lookup(square).type] if self.lookup(square).type in self.move_dict else []
        res = []
        for (dx,dy) in d:
            dire = []
            for i in range(max(self.H,self.W)):
                if not (0<=x+dx*(i+1)<self.W and 0<=y+dy*(i+1)<self.H):  break
                dire.append((x+dx*(i+1),y+dy*(i+1)))
                if self.lookup((x+dx*(i+1),y+dy*(i+1))) is not None: break
            res.append(dire)
        return res
    def validmove(self, square1: str | tuple, square2: str | tuple,color: int) -> bool:
        """Returns whether move would put king in check."""
        piece1, piece2 = self.lookup(square1),self.lookup(square2)
        del self[square1]
        self[square2] = piece1
        king = self.white_king if color==1 else self.black_king
        res = not self.check(king)
        self[square1] = piece1
        self[square2] = piece2
        return res
    def save(self, file: str = 'board.txt', stats: list = []) -> bool:
        """Save board state to text file. Stats also saves info: []."""
        board = self.simplified()+'\n'
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),file), 'w') as f:
            f.write(board)
            if stats: f.write('!'+' '.join([str(s) for s in stats]))
        return True
    def load(self, file: str = 'board.txt') -> bool | list:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),file), 'r') as f:
                D = dict([(v,k) for k,v in self.simplified_dict.items()])
                for y,line in enumerate(f.readlines()):
                    if line.startswith('!'):
                        return line[1:].split()
                    for x,c in enumerate(line.strip()):
                        P = D[c.lower()] if c.lower() in D else None
                        if P is None: self[(x,y)] = None
                        else: self[(x,y)] = Piece(P, 1 if c.isupper() else 0,self)
            return True
        except: return False
    def show(self, color: int = 1, info: bool = True, RICH: bool = True):
        """Prints board with correct orientation."""
        LETTERS = "".join([chr(97+i) for i in range(self.W)])
        if color == 0: LETTERS = LETTERS[::-1]
        letters = ''.join([c.center(self.gap) for c in LETTERS]) if info else ''
        res = []
        board = self.board[::-1] if color == 1 else [line[::-1] for line in self.board]
        for i,line in enumerate(board):
            res.append("|")
            if info: S = str(self.H-i) if color == 1 else str(i+1)
            else: S = ''
            res[i] += ''.join([(f"[{WHITE if p.color else BLACK}]" if RICH else '')+str(p).center(self.gap)+(f"[/{WHITE if p.color else BLACK}]" if RICH else '') if p is not None else (f"[{EMPTY}]" if RICH else '')+self.blank.center(self.gap)+(f"[/{EMPTY}]" if RICH else '') for p in line])+"| "+S
        return "\n".join(res)+'\n '+letters
    def lookup(self, notation: str | tuple):
        """Returns the piece on the square or none."""
        x,y = self.ind(notation)
        return self.board[y][x]
    def ind(self, notation: str) -> tuple:
        "Chess notation -> List index (x,y)"
        if type(notation) == tuple: return notation
        x,y = notation
        x = "abcdefgh".index(x)
        y = int(y)-1
        return (x,y)
    def code(self, index: tuple) -> str:
        """List index (x,y) -> Chess Notation"""
        if type(index) == str: return index
        x,y = index
        return chr(x+97)+str(y+1)
    def simplified(self) -> str:
        """Returns a simplified version of the board."""
        res = []
        blank = self.simplified_dict[None]
        for line in self.board:
            res.append(''.join([p.simplified if p is not None else blank for p in line]))
        return "\n".join(res)
    def setup(self) -> bool:
        """Sets up board with correct pieces."""
        if (self.W,self.H) != (8,8): return False
        def P(type, color): return Piece(type, color, self)
        black_back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
        white_back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
        for col, t in enumerate(black_back): self.board[0][col] = P(t, 1)
        for col, t in enumerate(white_back): self.board[7][col] = P(t, 0)
        for col in range(8): 
            self.board[1][col] = P("pawn", 1)
            self.board[6][col] = P("pawn", 0)
        return True
    def clear(self):
        """Removes all pieces."""
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
    def move(self,square1: str | tuple, square2: str | tuple) -> bool:
        """Move piece to chosen square. Requires square to be empty."""
        (x1,y1),(x2,y2) = self.ind(square1),self.ind(square2)
        if self.lookup((x2,y2)) is None:
            self.board[y2][x2] = self.board[y1][x1]
            self.board[y1][x1] = None
            if (p:=self.lookup((x2,y2))).type == 'king':
                if p.color == 1: self.white_king = self.code((x2,y2))
                elif p.color == 0: self.black_king = self.code((x2,y2))
            return True
        else: return False
    def capture(self, square1: str | tuple | None, square2: str | tuple) -> bool:
        """Move piece and capture square. Requires square to have a piece."""
        if square1 is None:
            x2,y2 = self.ind(square2)
            if self.lookup((x2,y2)) is None: return False
            place = self.white_captured if self.board[y2][x2].color == 1 else self.black_captured
            place.add(self.board[y2][x2])
            self.board[y2][x2] = None
            return True
        (x1,y1),(x2,y2) = self.ind(square1),self.ind(square2)
        if self.lookup((x2,y2)) is not None:
            place = self.white_captured if self.board[y2][x2].color == 1 else self.black_captured
            place.add(self.board[y2][x2])
            self.board[y2][x2] = self.board[y1][x1]
            self.board[y1][x1] = None
            if (p:=self.lookup((x2,y2))).type == 'king':
                if p.color == 1: self.white_king = self.code((x2,y2))
                elif p.color == 0: self.black_king = self.code((x2,y2))
            return True
        else: return False
    def move_options(self, square: str | tuple, CHECK: bool = False) -> list:
        """Returns a list of all possible move options for square."""
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        moves = piece.move((x,y))
        if piece.type == 'king': moves = list(filter(lambda X: self.check_king_move((x,y),X),moves))
        elif CHECK: moves = list(filter(lambda X: self.validmove((x,y),X,piece.color),moves))
        return moves
    def convert(self, values: str | tuple | list) -> str | tuple | list:
        """Converts index to notation and viceversa."""
        if type(values) != list: return self.code(values) if type(values)==tuple else self.ind(values)
        return [self.code(v) if type(v)==tuple else self.ind(v) for v in values]
    def capture_options(self, square: str | tuple, opposite: bool = False, CHECK: bool = False) -> list:
        """Opposite treats piece as opposite color."""
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        moves = piece.capture((x,y),opposite)
        if piece.type == 'king': moves = list(filter(lambda X: self.check_king_move((x,y),X),moves))
        elif CHECK: moves = list(filter(lambda X: self.validmove((x,y),X,piece.color),moves))
        return moves
    def check_king_move(self, king: str | tuple, square: str | tuple) -> bool:
        """Returns whether king can move to square without going in check."""
        x,y = self.ind(square)
        piece = self.lookup(king)
        del b[king]
        res = not self.check((x,y),piece.color)
        b[king] = piece
        return res
    def advantage(self, color: int = 1) -> int:
        """Returns point advantage for color."""
        if color == 0: color = -1
        return color*(sum(p.value() for p in self.black_captured)-sum(p.value() for p in self.white_captured))
    def turn(self, stats: list) -> bool | list:
        """Turn of specified color. Returns whether to continue."""
        global RICH,WHITE,BLACK,EMPTY
        color, info, swap = stats
        reverse = color if swap else 1
        quitcheck = False
        print('\n'+self.show(reverse,bool(info),bool(RICH)))
        while True:
            try:
                move = input("> ").lower()
            except:
                print("Quitting...")
                quit()
            if move in ['h','help','format']:
                print("""Use chess notation or x,y pairs.
- A = Shows all available moves from square. Ex: a1
- A B = Moves piece in square to specified. Ex: 1,2 1,4
- quit | info | save/load (file) | swap | show | color (white/black/empty COLOR)""")
                continue
            elif move in ['stop','cancel','x','quit']: 
                print("Cancelled.")
                return False
            elif not move:
                if quitcheck: return False
                else:
                    print("Input nothing again to quit.")
                    quitcheck = True
                continue
            elif move in ['show','view']:
                print('\n'+self.show(reverse,bool(info),bool(RICH)))
                continue
            elif move.split()[0] == 'save':
                file = 'board.txt' if len(move.split())==1 else move.split()[1]
                if self.save(file,stats): print(f"Board saved to '{file}' successfully.")
                else: print("Error. Unable to save the board.")
                continue
            elif move.split()[0] == 'load':
                file = 'board.txt' if len(move.split())==1 else move.split()[1]
                new = self.load(file)
                if new: print("Board loaded successfully.")
                else:
                    print(f"Error loading the board. Check if {file} exists.")
                    return [color]
                return new
            elif move in ['i','info']:
                print(f"Toggled info {"OFF" if info else "OFF"}.")
                return [color, int(not info)]
            elif move == 'swap':
                print(f"Toggled swapping {"OFF" if swap else "ON"}.")
                return [color, info, int(not swap)]
            elif move.split()[0] in ['color','rich']:
                if len(move.split()) == 1:
                    print(f"Toggled colors {"OFF" if RICH else "ON"}.")
                    RICH = not RICH
                    return [color]
                elif len(move.split()) == 3:
                    which, col = move.split()[1:]
                    if which in ['w','white','1']: WHITE, old = col, WHITE
                    elif which in ['black','0','b']: BLACK, old = col, BLACK
                    elif which in ['blank','empty','2','e']: EMPTY, old = col, EMPTY
                    else: print("Can only change 'white' (1), 'black' (0) or 'empty' (2). Note some colors may not work, use the python rich database for info.")
                    try:
                        print('\n'+self.show(reverse,bool(info),bool(RICH)))
                        print("Color changed successfully. ")
                    except:
                        if which in ['w','white','1']: WHITE = old
                        elif which in ['black','0','b']: BLACK = old
                        elif which in ['blank','empty','2','e']: EMPTY = old
                        print("Invalid color.")
                    continue
                else:
                    print("Invalid color change. Type 'color' to toggle or 'color white/black/empty COLOR'.")
                    continue
            move = move.split()
            try:
                if len(move) == 1:
                    move = move[0]
                    if ',' in move:
                        x,y = move.split(',')
                        move = (int(x),int(y))
                    if self.lookup(move) is None:
                        print("Square is empty.")
                        continue
                    elif self.lookup(move).color != color:
                        print("Piece belongs to opponent.")
                        continue
                    print("Move options:",self.convert(self.move_options(move,CHECK=True)))
                    print("Capture options:",self.convert(self.capture_options(move,CHECK=True)))
                    continue
                elif len(move) == 2:
                    moveA,moveB = move
                    if ',' in moveA: 
                        x,y = moveA.split(',')
                        moveA = (int(x),int(y))
                    if ',' in moveB: 
                        x,y = moveB.split(',')
                        moveB = (int(x),int(y))
                    if self.lookup(moveA) is None:
                        print("Square is empty.")
                        continue
                    elif self.lookup(moveA).color != color:
                        print("Piece belongs to opponent.")
                        continue
                    opt, cap = self.move_options(moveA,CHECK=True), self.capture_options(moveA,CHECK=True)
                    if self.ind(moveB) in opt:
                        self.move(moveA,moveB)
                        if info: print(f"{self.lookup(moveB).type.capitalize()} moved to {self.code(moveB)}.")
                        break
                    elif self.ind(moveB) in cap:
                        capt = self.lookup(moveB).type.capitalize()
                        self.capture(moveA, moveB)
                        if info: print(f"{self.lookup(moveB).type.capitalize()} captured {capt} on {self.code(moveB)}.")
                        break
                    else:
                        print("Invalid move. Try typing the piece to see available moves.")
                else: print("Unknown input. Type 'help' for info.")
            except:
                print("Unknown input. Type 'help' for info.")
        if info: print(f"{"White" if color==1 else "Black"}'s material advantage: {self.advantage(color=color)}")
        return True
    
    def mainloop(self):
        """Full match. If reverse is False, only show board in main orientation.
        Still WIP, need to add checkmates, etc."""
        color = 1
        info = 1
        swap = 0
        while True:
            print(f"\n{"White" if color==1 else "Black"}'s turn.")
            stats = [color, info, swap]
            status = self.turn(stats)
            if status is False: break
            color = int(not color)
            if isinstance(status,list):
                for i,s in enumerate(status):
                    if i == 0: color = int(s)
                    if i == 1: info = int(s)
                    if i == 2: swap = int(s)
class Piece:
    def __init__(self, type: str, color: int, board: Board | None = None):
        """Chess piece. Color 0 = Black, 1 = White.
        Type options: pawn | knight | rook | bishop | queen | king"""
        self.type = type
        self.color = color
        self.board = board if board is not None else Board()
        self.symbol = self.board.symbol_dict[self.type] if self.type in self.board.symbol_dict else "?"
        self.simplified = self.board.simplified_dict[self.type] if self.type in self.board.simplified_dict else "?"
        if self.color == 1: self.simplified = self.simplified.capitalize()
    def __repr__(self):
        return f"Piece(type={self.type},color={'white' if self.color==1 else 'black' if self.color==0 else f'??? ({self.color})'})"
    def __str__(self):
        return self.symbol
    def value(self) -> int:
        """Returns value of piece."""
        return self.board.points[self.type]
    def options(self, location: tuple) -> list:
        """Returns all possible move options on empty board, as list of (x,y)."""
        x,y = location
        options = []
        N = max(self.board.H,self.board.W)
        d = self.board.move_dict[self.type] if self.type in self.board.move_dict else []
        match self.type:
            case 'pawn':
                if self.color==1:
                    options.append((0,1))
                    if y == 1: options.append((0,2))
                elif self.color==0:
                    options.append((0,-1))
                    if y == self.board.H-2: options.append((0,-2))
            case 'knight': options = d
            case 'rook' | 'bishop' | 'queen':
                for (dx,dy) in d:
                    for i in range(N):
                        if not (0<=x+dx*(i+1)<self.board.W and 0<=y+dy*(i+1)<self.board.H) or self.board.lookup((x+dx*(i+1),y+dy*(i+1))) is not None: 
                            break
                        options.append((dx*(i+1),dy*(i+1)))
            case 'king': options = d
            case _:
                return []
        options = list(filter(lambda a: 0<=a[0]<self.board.W and 0<=a[1]<self.board.H, [(x+dx,y+dy) for dx,dy in options]))
        return options
    def move(self, location: tuple) -> list:
        """Returns possible move options as list of (x,y)."""
        return list(filter(lambda x: self.board.lookup(x) is None,self.options(location)))
    def capture(self, location: tuple, opposite: bool = False) -> list:
        """Returns possible capture options as list of (x,y). Opposite treats piece as other color."""
        x,y = location
        options = []
        N = max(self.board.H,self.board.W)
        d = self.board.move_dict[self.type] if self.type in self.board.move_dict else []
        match self.type:
            case 'pawn':
                if self.color==1: options = [(1,1),(-1,1)]
                elif self.color==0: options = [(1,-1),(-1,-1)]
            case 'knight': options = d
            case 'rook' | 'bishop' | 'queen':
                for (dx,dy) in d:
                    for i in range(N):
                        if not (0<=x+dx*(i+1)<self.board.W and 0<=y+dy*(i+1)<self.board.H):
                            break
                        options.append((dx*(i+1),dy*(i+1)))
                        if self.board.lookup((x+dx*(i+1),y+dy*(i+1))) is not None: 
                            break
            case 'king': options = d
            case _:
                return []
        C = lambda X: self.board.lookup(X).color == self.color if opposite else self.board.lookup(X).color != self.color
        return list(filter(lambda x: 0<=x[0]<self.board.W and 0<=x[1]<self.board.H and self.board.lookup(x) is not None and C(x),[(x+dx,y+dy) for dx,dy in options]))

if __name__ == '__main__':
    b = Board(filled=True)
    print("WIP chess.")
    b.mainloop()
