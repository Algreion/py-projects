
pprint = print
from rich import print
from random import choice,randint
import os,pygame

RICH = True # Colors
BLACK = 'cyan'
WHITE = 'cornsilk1'
EMPTY = 'bold'

LOGS = True
SEP = "_______________________________"

ADMIN = True
DEBUGGING = False

class Board:
    def __init__(self, N: tuple = (8,8), filled: bool = False, type: str | None = 'Chess'):
        """Chessboard."""
        self.W, self.H = N
        self.type = type
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
        self.symbol_dict = {"king":"♔","queen":"♕","rook":"♖","bishop":"♗","knight":"♘","pawn":"♙"}
        self.simplified_dict = {"king":"k","queen":"q","rook":"r","bishop":"b","knight":"n","pawn":"p",None: '.'}
        self.simplified_dict_invert = dict([(v,k) for k,v in self.simplified_dict.items()])
        self.gap = 2 # board str
        self.blank = '▢'
        self.whitemoved = [True,True,True]
        self.blackmoved = [True,True,True]
        self.white_king = None
        self.black_king = None
        self.default = self.implicit
        self.white_enpassant,self.black_enpassant = None, None
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
    def view(self, color: int = 1, info: bool = True) -> None:
        """Prints the board."""
        print(self.show(color, info, RICH))
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
                    if piece is None or piece.color == color: T = True
                    else: T = False
                    if self.ind(square) in self.capture_options((w,h),override=T)+self.move_options((w,h)):
                        if pieces: attacking.append((w,h))
                        else: return True
        return attacking if pieces else False
    def stalemate(self, color: int = 1) -> bool:
        """Flag of stalemate. (No legal moves, king not in check.)"""
        for h in range(self.H):
            for w in range(self.W):
                p = self.lookup((w,h)) 
                if p is not None and p.color == color and self.move_options((w,h),CHECK=True,CASTLING=True)+self.capture_options((w,h),CHECK=True): return False
        return not self.checkcheck(color)
    def drawmate(self) -> bool:
        """Returns whether only kings are left on the board."""
        for h in range(self.H):
            for w in range(self.W):
                p = self.lookup((w,h))
                if p is not None and p.type != 'king': return False
        return True
    def checkmate(self, color: int = 1) -> bool:
        """Flag of whether the king is in checkmate."""
        if color == 1: king = self.white_king
        else: king = self.black_king
        attacking = self.check(king,pieces=True)
        if not attacking: return False
        for o in self.capture_options(king,CHECK=True)+self.move_options(king,CHECK=True):
            if not self.check(o,color): return False
        if len(attacking) == 1:
            xp,yp = attacking[0]
            xk,yk = self.ind(king)
            defending = self.check((xp,yp),pieces=True)
            if defending:
                for p in defending:
                    if self.validmove(p,(xp,yp),color): return False
            if self.lookup((xp,yp)).type in ['rook','queen','bishop']:
                dirs = self.directions((xp,yp))
                opts = []
                for direction in dirs:
                    if (xk,yk) in direction:
                        opts = direction
                        break
                for w in range(self.W):
                    for h in range(self.H):
                        if (P:=self.lookup((w,h))) is not None and P.color == color:
                            if any([move in opts for move in self.move_options((w,h))]): return False
        return True
    def setpiece(self, square: str | tuple, name: str | None = None, color: int | None = None):
        """Creates or modifies an existing piece."""
        piece = self.lookup(square)
        if piece is None: name, color = name if name is not None else 'pawn', color if color is not None else 1
        else: name, color = name if name is not None else piece.type, color if color is not None else piece.color
        self[square] = Piece(name, color, self)
        if name == 'king':
            if color == 1: self.white_king = square
            else: self.black_king = square
    def checkcheck(self, color: int) -> bool:
        """Returns whether king is in check."""
        king = self.white_king if color == 1 else self.black_king
        return self.check(king)
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
    def validmove(self, square1: str | tuple, square2: str | tuple, color: int, enpassant: bool = False) -> bool:
        """Returns whether move would put king in check."""
        piece1, piece2 = self.lookup(square1),self.lookup(square2)
        if enpassant:
            del self[square1]
            self[square2] = piece1
            x,y = self.ind(square2)
            if color == 1: y -= 1
            else: y += 1
            piece = self.lookup((x,y))
            del self[(x,y)]
            res = not self.checkcheck(color)
            self[square1] = piece1
            self[square2] = piece2
            self[(x,y)] = piece
        else:
            del self[square1]
            self[square2] = piece1
            if piece1 is not None and piece1.type == 'king': king = square2
            else: king = self.white_king if color==1 else self.black_king
            res = not self.check(king)
            self[square1] = piece1
            self[square2] = piece2
        return res
    def save(self, file: str = 'board.txt', stats: list = []) -> bool:
        """Save board state to text file. Stats also saves info: [Turn, Info, Swap]."""
        board = self.simplified()+'\n'
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),file), 'w') as f:
            f.write(board)
            f.write('='+','.join([str(int(c)) for c in self.whitemoved])+' '+','.join([str(int(c)) for c in self.blackmoved]))
            f.write('\n&'+','.join(['' if c is None else self.code(c) for c in [self.white_enpassant,self.black_enpassant]]))
            if stats: f.write('\n!'+' '.join([str(s) for s in stats]))
        return True
    def load(self, file: str = 'board.txt') -> bool | list:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),file), 'r') as f:
                D = self.simplified_dict_invert
                for y,line in enumerate(f.readlines()):
                    if line.startswith('='): # Castling check
                        self.whitemoved,self.blackmoved = [bool(int(i)) for i in line[1:].split()[0].split(',')],[bool(int(i)) for i in line[1:].split()[1].split(',')]
                        continue
                    elif line.startswith('&'):  # En Passant check
                        self.white_enpassant,self.black_enpassant = [c if c else None for c in line[1:].strip().split(',')]
                        continue
                    if line.startswith('!'):
                        return line[1:].split()
                    for x,c in enumerate(line.strip()):
                        P = D[c.lower()] if c.lower() in D else None
                        if P is None: self[(x,y)] = None
                        else:
                            color = 1 if c.isupper() else 0
                            self[(x,y)] = Piece(P, color,self)
                            if P == 'king' and color == 1: self.white_king = self.code((x,y))
                            elif P == 'king' and color == 0: self.black_king = self.code((x,y))
            return True
        except Exception as e: 
            if DEBUGGING: print(f"[Debug] Error: {e}")
            return False
    def show(self, color: int = 1, info: bool = True, RICH: bool = True) -> str:
        """Returns string of board with correct orientation."""
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
        try:
            x,y = notation
            x = "abcdefgh".index(x)
            y = int(y)-1
            return (x,y)
        except: return notation
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
        self.clear()
        if (self.W,self.H) != (8,8): return False
        self.whitemoved = [False,False,False]
        self.blackmoved = [False,False,False]
        self.white_king = 'e1'
        self.black_king = 'e8'
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
        self.whitemoved = [True,True,True]
        self.blackmoved = [True,True,True]
        self.white_king = None
        self.black_king = None
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
    def implicit(self, move: str) -> str:
        """Converts explicit (b1 c3) to implicit (Nc3)"""
        try:
            if len(move) != 5 or ' ' not in move: return move
            square1, square2 = move[:2],move[3:]
            piece = self.lookup((square1))
            if piece is None:
                if DEBUGGING: print(f"[Debug] Conversion of '{move}' to implicit failed. Square is empty.")
                return move
            p = self.simplified_dict[piece.type].upper()
            return f"{p}{square2}"
        except Exception as e: 
            if DEBUGGING: print(f"[Debug] Conversion of '{move}' to implicit failed. Error: {e}")
            return move
    def explicit(self, move: str, color: int | None = None, debugging: bool = False) -> str | None:
        """Converts implicit (Nc3) to explicit (b1 c3)"""
        try:
            if len(move) != 3: return move
            move = move.lower()
            P, square2 = move[0],move[1:]
            pieces,option = [], self.ind(square2)
            for h in range(self.H):
                for w in range(self.W):
                    p = self.lookup((w,h))
                    if p is not None and self.simplified_dict_invert[P] == p.type and option in self.move_options((w,h),True,True)+self.capture_options((w,h),CHECK=True):
                        if color is None or p.color==color: pieces.append(self.code((w,h)))
            if len(pieces) == 0:
                if debugging: print("Move isn't an option.")
                return None
            elif len(pieces) == 1:
                return f"{pieces[0]} {square2}"
            else:
                while True:
                    try: x = input(f"Ambiguous move, applies to pieces on: {pieces}. Specify square: ")
                    except: (LOGS and pprint("Quitting...")) or quit()
                    if x in pieces: return f"{x} {square2}"
                    elif x.isdigit() and 0<int(x)<=len(pieces): return f"{pieces[int(x)-1]} {square2}"
                    print("Invalid choice.")
        except Exception as e:
            if debugging: print(f"[Debug] Conversion of '{move}' to explicit failed. Error: {e}")
            return None
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
    def move_options(self, square: str | tuple, CHECK: bool = False, CASTLING: bool = False) -> list:
        """Returns a list of all possible move options for square. Note castling is handled in self.handle_turn"""
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        moves = piece.move((x,y))
        if piece.type == 'king':
            moves = list(filter(lambda X: self.check_king_move((x,y),X),moves))
            if CASTLING and self.castle(piece.color,long=False,commit=False): moves += ['castle']
            if CASTLING and self.castle(piece.color,long=True,commit=False): moves += ['longcastle']
        elif CHECK: moves = list(filter(lambda X: self.validmove((x,y),X,piece.color),moves))
        return moves
    def convert(self, values: str | tuple | list) -> str | tuple | list:
        """Converts index to notation and viceversa."""
        if type(values) != list: return self.code(values) if type(values)==tuple else self.ind(values)
        return [self.code(v) if type(v)==tuple else self.ind(v) for v in values]
    def capture_options(self, square: str | tuple, override: bool = False, CHECK: bool = False) -> list:
        """Override doesn't consider whether a piece is present."""
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        moves = piece.capture((x,y),override=override)
        if CHECK: moves = list(filter(lambda X: self.validmove((x,y),X,piece.color,enpassant=self.lookup(X) is None),moves))
        return moves
    def check_king_move(self, king: str | tuple, square: str | tuple) -> bool:
        """Returns whether king can move to square without going in check."""
        x,y = self.ind(square)
        piece = self.lookup(king)
        del self[king]
        res = not self.check((x,y),piece.color)
        self[king] = piece
        return res
    def getpieces(self, color: int | None = None) -> list:
        """Returns a list of all pieces of specified color (or both if None)."""
        F = lambda X: X is not None and (True if color is None else X.color == color)
        return list(filter(F,[p for y in range(self.H) for p in self.board[y]]))
    def advantage(self, color: int = 1) -> int:
        """Returns point advantage for color."""
        if color == 0: color = -1
        return color*(sum(p.value() for p in self.getpieces(1))-sum(p.value() for p in self.getpieces(0)))
    def castle(self, color: int, long: bool = False, commit: bool = False) -> bool:
        """Castles and returns if move was successful."""
        check = self.whitemoved if color==1 else self.blackmoved
        if color == 1:
            C = 'e1'
            if long: i,A,B,D,E = 0,'a1','c1','b1',self.lookup('d1') is None
            else: i,A,B,D,E = 2,'h1','f1','g1',True
        else:
            C = 'e8'
            if long: i,A,B,D,E = 0,'a8','c8','b8',self.lookup('d8') is None
            else: i,A,B,D,E = 2,'h8','f8','g8',True
        if check[1] or check[i] or not (self.lookup(B) is None and self.lookup(D) is None and E):
            if commit: print("Unable to castle.")
            return False
        self.move(A,B)
        self.move(C,D)
        if self.check(D):
            self.move(B,A)
            self.move(D,C)
            if commit: print("Unable to castle.")
            return False
        if commit: 
            if color == 1: self.whitemoved = (True,True,True)
            else: self.blackmoved = (True,True,True)
            return True
        self.move(B,A)
        self.move(D,C)
        return True
    def checkpromo(self, square: str | tuple) -> bool:
        """Returns whether pawn can be promoted."""
        square, piece = self.ind(square), self.lookup(square)
        return piece.type == 'pawn' and ((piece.color == 1 and square[1] == 7) or (piece.color == 0 and square[1] == 0))
    def all_moves(self, color: int, capturesOnly: bool = False, movesOnly: bool = False) -> tuple:
        """Returns a list of all possible (moves, captures) for the color. 'Only' options save time by only checking that."""
        pieces = []
        for w in range(self.W):
            for h in range(self.H):
                if self.lookup((w,h)) is not None and self.lookup((w,h)).color == color: pieces.append((w,h))
        moves = []
        captures = []
        for p in pieces:
            if not capturesOnly: moves += [f"{self.code(p)} {self.code(move)}" if move not in ['castle','longcastle'] else move for move in self.move_options(p,CHECK=True,CASTLING=True)]
            if not movesOnly: captures += [f"{self.code(p)} {self.code(move)}" for move in self.capture_options(p,CHECK=True)]
        return moves,captures
    def ai(self, color: int, logic: int = 1) -> str | None:
        """Returns a move based on chosen logic. Returns None if no available moves.
        Logic 1: Completely random.
        Logic 2: Prefers captures. Still random.
        Logic 3: Avoids captures at all costs."""
        match logic:
            case 1:
                pool = self.all_moves(color)
                pool = pool[0]+pool[1]
                if not pool: return None
                return choice(pool)
            case 2:
                pool = self.all_moves(color)
                if pool[1]: return choice(pool[1])
                if not pool[0]: return None
                return choice(pool[0])
            case _:
                pool = self.all_moves(color)
                if pool[0]: return choice(pool[0])
                if not pool[1]: return None
                return choice(pool[1])
    def handle_turn(self, move: str, color: int) -> bool | str:
        """Handles game move with correct format. Returns True=break, False=continue, str=pawn promotion square."""
        global RICH,WHITE,BLACK,EMPTY,LOGS
        move = move.split()
        if len(move) == 1:
            move = move[0]
            if move in ['c','castle']:
                res = self.castle(color=color,long=False,commit=True)
                if res: 
                    if LOGS: print("Castled.")
                    return True
                else: return False
            elif move in ['cl','lc','longcastle']:
                res = self.castle(color=color,long=True,commit=True)
                if res: 
                    if LOGS: print("Castled queen side.")
                    return True
                else: return False
            if ',' in move:
                x,y = move.split(',')
                move = (int(x),int(y))
            if self.lookup(move) is None:
                test = self.explicit('P'+move,color)
                if test != move and test is not None:
                    self.handle_turn(test,color)
                    return True
                print("Square is empty.")
                return False
            elif self.lookup(move).color != color:
                print("Piece belongs to opponent.")
                return False
            A,B = self.convert(self.move_options(move,CHECK=True,CASTLING=True)), self.convert(self.capture_options(move,CHECK=True))
            print(f"Move options: {A} ({len(A)})")
            print(f"Capture options: {B} ({len(B)})\n")
            return False
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
                return False
            elif self.lookup(moveA).color != color:
                print("Piece belongs to opponent.")
                return False
            opt, cap = self.move_options(moveA,CHECK=True), self.capture_options(moveA,CHECK=True)
            if self.ind(moveB) in opt:
                self.move(moveA,moveB)
                self.whitemoved,self.blackmoved = self.helper_moved(moveA,self.whitemoved,self.blackmoved)
                if LOGS: print(f"{"White" if color == 1 else "Black"} {self.lookup(moveB).type.capitalize()} moved to {self.code(moveB)}.")
                if self.lookup(moveB).type == 'pawn' and abs(self.ind(moveB)[1]-self.ind(moveA)[1]) == 2:
                    x,y = self.ind(moveB)
                    if color == 1: self.white_enpassant = self.ind((x,y-1))
                    else: self.black_enpassant = self.ind((x,y+1))
                else:
                    if color == 1: self.white_enpassant = None
                    else: self.black_enpassant = None
                if self.checkpromo(moveB): return moveB
                return True
            elif self.ind(moveB) in cap:
                if self.lookup(moveB) is None:
                    x,y = self.ind(moveB)
                    if color == 1: y -= 1
                    else: y += 1
                    capt = self.lookup((x,y)).type.capitalize()
                    self.capture(moveA, (x,y))
                    self.move((x,y),moveB)
                    if LOGS: print(f"{"White" if color == 1 else "Black"} {self.lookup(moveB).type.capitalize()} captured {capt} on {self.code(moveB)}. (En passant)")
                else:
                    capt = self.lookup(moveB).type.capitalize()
                    self.capture(moveA, moveB)
                    self.whitemoved,self.blackmoved = self.helper_moved(moveA,self.whitemoved,self.blackmoved)
                    if LOGS: print(f"{self.lookup(moveB).type.capitalize()} captured {capt} on {self.code(moveB)}.")
                self.white_enpassant,self.black_enpassant = None, None
                if self.checkpromo(moveB): return moveB
                return True
            else:
                print("Invalid move. Try typing the piece to see available moves.")
        else:
            print("Unknown input. Type 'help' for info.")
    def helper_moved(self, move: str, whitemoved: list,blackmoved: list) -> tuple:
        """Returns (whitemoved, blackmoved)"""
        match move:
            case 'e1': self.whitemoved[1] = True
            case 'a1': self.whitemoved[0] = True
            case 'h1': self.whitemoved[2] = True
            case 'e8': self.blackmoved[1] = True
            case 'a8': self.blackmoved[0] = True
            case 'h8': self.blackmoved[2] = True
        return whitemoved,blackmoved
    def turn(self, stats: list, botplay: tuple | None = None) -> bool | list | int:
        """Turn of specified color. Returns whether to continue. Pawn promotions handled here due to input requirement."""
        global RICH,WHITE,BLACK,EMPTY,LOGS
        color, info, swap = stats
        reverse = color if swap else 1
        quitcheck = False
        asked_draw = False
        if botplay and botplay[0] == color: 
            botcolor, botlogic = botplay
            move = self.ai(botcolor,botlogic)
            if move is None:
                pprint("Bot couldn't find a valid move. Quitting...")
                quit()
            outcome = self.handle_turn(move,botcolor)
            if isinstance(outcome,str):
                match botlogic:
                    case 1: options = ['queen','bishop','rook','knight'] # Medium
                    case 2: options = ['queen'] # Hard
                    case 3: options = ['bishop','knight'] # Easy
                    case _: options = ['rook'] # Misc
                promo = choice(options)
                self.setpiece(outcome, name=promo)
                if LOGS: print(f"{"WHITE" if color == 1 else "Black"} promoted pawn on {outcome} to a {promo.capitalize()}.")
            return True
        print('\n'+self.show(reverse,bool(info),bool(RICH)))
        if self.checkcheck(color):
            if LOGS: print("Your king is in check!")
            0
        while True:
            try:
                if RICH: print(f"{f"[{WHITE}][White][/{WHITE}]" if color==1 else f"[{BLACK}][Black][/{BLACK}]"} ",end='')
                else: print(f"[{"White" if color==0 else "Black"}]",end='')
                move = input("> ").lower()
            except:
                pprint("Quitting...")
                quit()
            if move in ['h','help','format','cmds']:
                print("""Use chess notation (eg. Nc3, e4, Qh7, Ph7...) or square notation:
- A = Shows all available moves from square. Ex: a1
- A B = Moves piece in square to specified. Ex: a2 a4
Chess commands: castle | draw | surrender || Commands: info | save (file) | swap | show | logs | color (white/black/empty COLOR)\n""")
                if ADMIN: print("""Admin commands:\nload (file) | reset | all (moves/captures) | random (1-3) | debug | default\nwin | forcedraw | set SQUARE PIECE (COLOR) | del SQUARE\n""")
                continue
            elif move in ['surrender','resign','giveup','stop','cancel','x','quit']: 
                print(f"{"White" if color==1 else "Black"} resigns.")
                return int(not color)
            elif move in ['draw','tie']:
                if asked_draw:
                    print("Already offered a draw this round.\n")
                    continue
                print(f"{"White" if color==1 else "Black"} offers a draw.\n")
                asked_draw = True
                if botplay: 
                    print("The bot refused.\n")
                    continue
                else:
                    print(f"{"White" if color==0 else "Black"}, accept (1) or refuse (2) the draw?")
                    try:
                        if RICH: print(f"{f"[{WHITE}][White][/{WHITE}]" if color==0 else f"[{BLACK}][Black][/{BLACK}]"} ",end='')
                        else: print(f"[{"White" if color==0 else "Black"}]",end='')
                        draw = input("> ").lower()
                        if draw in ['1','y','yes','accept']:
                            print(f"{"White" if color==0 else "Black"} accepted the offer.\n")
                            return -1
                        else:
                            print(f"{"White" if color==0 else "Black"} refused.\n")
                            continue
                    except:
                        pprint("Quitting...")
                        quit()
            elif not move:
                if quitcheck: return False
                else:
                    print("Input nothing again to quit.")
                    quitcheck = True
                continue
            elif move in ['def', 'default'] and ADMIN:
                if self.default == self.implicit:
                    print("Default move rendering is now explicit. Ex: 'c1 b3'.")
                    self.default = self.explicit
                elif self.default == self.explicit:
                    print("Default move rendering is now implicit. Ex: 'Nc3'.")
                    self.default = self.implicit
                continue
            elif move in ['show','view']:
                print('\n'+self.show(reverse,bool(info),bool(RICH)))
                continue
            elif move.split()[0] in ['save','export']:
                file = 'board.txt' if len(move.split())==1 else move.split()[1]
                if self.save(file,stats): print(f"Board saved to '{file}' successfully.\n")
                else: print("Error. Unable to save the board.")
                continue
            elif move.split()[0] in ['load','import'] and ADMIN:
                file = 'board.txt' if len(move.split())==1 else move.split()[1]
                new = self.load(file)
                if new: print("Board loaded successfully.")
                else:
                    print(f"Error loading the board. Check if {file} exists.")
                    return [color]
                return new
            elif move in ['i','info']:
                print(f"Toggled info {"OFF" if info else "ON"}.")
                return [color, int(not info)]
            elif move == 'logs':
                print(f"Toggled logs {"OFF" if LOGS else "ON"}.")
                LOGS = not LOGS
                return [color]
            elif move == 'swap':
                print(f"Toggled swapping {"OFF" if swap else "ON"}.")
                return [color, info, int(not swap)]
            elif move.split()[0] == 'all' and ADMIN:
                move = move.split()
                if len(move) == 1:
                    pool = self.all_moves(color)
                    pool1, pool2 = pool
                    if self.default == self.implicit: pool1, pool2 = [self.implicit(i) for i in pool1],[self.implicit(i) for i in pool2]
                    print(f"All available moves:\n{pool1}\nAll available captures:\n{pool2}")
                elif move[1] in ['c','captures']:
                    pool = self.all_moves(color,capturesOnly=True)[1]
                    if self.default == self.implicit: pool = [self.implicit(i) for i in pool]
                    print(f"All available captures:\n{pool}")
                elif move[1] in ['m','moves']: 
                    pool = self.all_moves(color,movesOnly=True)[0]
                    if self.default == self.implicit: pool = [self.implicit(i) for i in pool]
                    print(f"All available moves (excluding captures):\n{pool}")
                else: print("Invalid input. Options are 'all', 'all moves' or 'all captures'.")
                continue
            elif move.split()[0] in ['rand','random','r'] and ADMIN:
                if len(move) == 1: logic = 2
                else:
                    if not move.split()[1].isdigit():
                        print("Logic must be a number.")
                    logic = int(move.split()[1])
                m = self.ai(color,logic)
                if m is None: 
                    print("No valid move found.")
                    continue
                else:
                    outcome = self.handle_turn(move=m,color=color)
                    if outcome: break
                    else: continue
            elif move in ['points','advantage','material','mat','adv','eval']:
                print(f"{f"[{WHITE}]White[/{WHITE}]" if color==1 else f"[{BLACK}]Black[/{BLACK}]"}'s material advantage: {self.advantage(color=color)}")
                continue
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
            elif ADMIN and move.split()[0] in ['win','forcedraw','fdraw','set','del','reset','debug']:
                move = move.split()
                try:
                    match move[0]:
                        case 'win': return color
                        case 'forcedraw' | 'fdraw': return -1
                        case 'set':
                            square, name = move[1],move[2]
                            col = color if len(move) == 3 else move[3]
                            if col in ['w','white']: col = 1
                            elif col in ['b','black']: col = 0
                            self.setpiece(square,name,col)
                            print(f"Set piece on {square} to {"White" if col==1 else "Black"} {name.capitalize()}.")
                            print("\n")
                            self.view()
                        case 'del':
                            square = move[1]
                            p = self.lookup(square)
                            if p is None: print(f"No piece on {square}.")
                            else: print(f"Deleted {p.type.capitalize()} on {square}.")
                            del self[square]
                            print("\n")
                            self.view()
                        case 'reset':
                            self.clear()
                            self.setup()
                            print("\n")
                            self.view()
                        case 'debug': print(f"whitemoved: {self.whitemoved} | blackmoved: {self.blackmoved} (a8,e8,h8)\nWhite King: {self.white_king} | Black King: {self.black_king}\nCaptured white: [{','.join(str(p) for p in self.white_captured)}] | Captured black: [{','.join(str(p) for p in self.black_captured)}]\nWhite enpassant: {self.white_enpassant} | Black enpassant: {self.black_enpassant}\nWhite eval: {self.advantage()} | Color={color}, Info={info}, Swap={swap}, Bot={botplay}\n")
                except Exception as e:
                    print("Invalid command usage. Type 'help' for info.")
                    if DEBUGGING: print(f"[Debug] Error: {e}")
                continue
            try:
                move = self.explicit(move,color)
                if move is None:
                    print("Illegal move.\n")
                    continue
                outcome = self.handle_turn(move=move,color=color)
                if isinstance(outcome,str): # Pawn promotion
                    print("\n")
                    self.view(color, info)
                    pprint(f"\nPawn at {outcome} can be promoted: ''/1/Q = Queen | 2/N = Knight | 3/R = Rook | 4/B = Bishop")
                    if RICH: print(f"{f"[{WHITE}][White][/{WHITE}]" if color==1 else f"[{BLACK}][Black][/{BLACK}]"} ",end='')
                    try: promo = input("> ")
                    except: quit()
                    match promo:
                        case '2' | 'n': promo = 'knight'
                        case '3' | 'r': promo = 'rook'
                        case '4' | 'b': promo = 'bishop'
                        case _: promo = 'queen'
                    self.setpiece(outcome, name=promo)
                    if LOGS: print(f"Promoted pawn on {outcome} to a {promo.capitalize()}.")
                if outcome: break
                else: continue
            except Exception as e:
                print("Unknown input. Type 'help' for info.")
                if DEBUGGING: print(f"[Admin Log]: Error | {e}\n")
        return True
    
    def mainloop(self, botplay: tuple | None = None) -> int:
        """Full match. If reverse is False, only show board in main orientation. Handles checkmate & stalemate. Returns who won, or -1 for tie."""
        color = 1
        info = 1
        swap = 0
        while True:
            print(SEP)
            print(f"\n{f"[{WHITE}]White[/{WHITE}]" if color==1 else f"[{BLACK}]Black[/{BLACK}]"}'s turn.")
            stats = [color, info, swap]
            status = self.turn(stats, botplay=botplay)
            if status is False: break
            elif type(status)==int:
                if status == -1: print("The match ends in a draw.")
                else: print(f"{f"[{WHITE}]White[/{WHITE}]" if status==1 else f"[{BLACK}]Black[/{BLACK}]"} wins!\n")
                return status
            color = int(not color)
            if self.stalemate(color) or self.drawmate():
                print(SEP)
                self.view(1,info)
                print(f"[bold]Stalemate![/bold]")
                return -1
            if self.checkmate(color):
                print(SEP)
                self.view(1,info)
                print(f"\n[bold]Checkmate![/bold] {f"[{WHITE}]White[/{WHITE}]" if color==0 else f"[{BLACK}]Black[/{BLACK}]"} has won.")
                return int(not color)
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
        return self.board.points[self.type] if self.type in self.board.points else 0
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
    def capture(self, location: tuple, override: bool = False) -> list:
        """Returns possible capture options as list of (x,y). Override doesn't consider whether a piece is present."""
        x,y = location
        options = []
        enpassant = []
        N = max(self.board.H,self.board.W)
        d = self.board.move_dict[self.type] if self.type in self.board.move_dict else []
        match self.type:
            case 'pawn':
                if self.color==1: 
                    options = [(1,1),(-1,1)]
                    if self.board.black_enpassant is not None and self.board.ind(self.board.black_enpassant) in [(x+dx,y+dy) for dx,dy in options]:
                        enpassant.append(self.board.ind(self.board.black_enpassant))
                elif self.color==0: 
                    options = [(1,-1),(-1,-1)]
                    if self.board.white_enpassant is not None and self.board.ind(self.board.white_enpassant) in [(x+dx,y+dy) for dx,dy in options]:
                        enpassant.append(self.board.ind(self.board.white_enpassant))
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
        C = lambda X: True if override else (self.board.lookup(X) is not None and self.board.lookup(X).color != self.color)
        return enpassant+list(filter(lambda x: 0<=x[0]<self.board.W and 0<=x[1]<self.board.H and C(x),[(x+dx,y+dy) for dx,dy in options]))

def superloop():
    """Full terminal experience."""
    global ADMIN,DEBUG
    HELP = "Options:\n1. Player vs player. | 2. Player vs bot. | 3. Load previous game. | 4. Stats. | 0. Quit. || Cmds: 'admin', 'debug'"
    print("[bold underline]PyChess[/bold underline]\n")
    pprint(HELP)
    white, black, total, botwins = 0,0,0,0
    quitcheck = False
    while True:
        try: choice = input("\n[PyChess] > ").lower()
        except:
            pprint("Quitting...")
            break
        match choice:
            case '1' | 'pvp':
                board = Board(filled=True)
                result = board.mainloop()
                if result == 1: white += 1
                elif result == 0: black += 1
                else:
                    white += 1
                    black += 1
                total += 1
            case '2' | 'bot':
                try: logic = input("Choose difficulty:\n1. Easy | 2. Medium | 3. Hard\n> ")
                except:
                    print("Quitting...")
                    break
                match logic:
                    case '1': logic = 3 # No captures
                    case '2': logic = 1 # Random
                    case '3': logic = 2 # Captures
                    case 'q' | 'x':
                        print("Cancelled")
                        continue
                    case _:
                        print("Invallid difficulty. Defaulting to medium.")
                        logic = 1
                color = input("\nChoose your color, or leave blank to pick white:\n1. White | 2. Black | 3. Random\n> ")
                match color:
                    case '1' | '': color = 0 # Color of bot
                    case '2': color = 1
                    case '3': 
                        color = randint(0,1)
                        pprint(f"Randomly assigned to {'White' if color == 0 else 'Black'}.")
                diff = 'Easy' if logic == 3 else 'Medium' if logic == 1 else 'Hard'
                board = Board(filled=True)
                if board.mainloop(botplay=(color,logic)) != color: 
                    if logic == 3: remark = "You just dashed the bot's hopes and dreams. Perhaps try a harder difficulty for a real challenge?"
                    elif logic == 1: remark = "You are officially better at chess than random chance. That puts you in the top 50% players... What do you mean that's not how statistics work?"
                    elif logic == 2: remark = "Good job! You just beat a bot at the level of Magnus Carlsen. You don't believe me? Well, it tried its best, and that's what counts."
                    print(f"[bold]Congratulations, you won against the Bot on {diff} difficulty![/bold]")
                    pprint(remark)
                    if color == 0: white += 1
                    else: black += 1
                    botwins += 1
                else: 
                    if logic == 3: remark = "That is... impressive, it should be impossible for it to win.\nYour chess skills are below zero, but I'm sure you have other talents!"
                    elif logic == 1: remark = "Fun fact, this bot plays completely at random. So don't feel bad for losing, there was a nonzero chance it played like a Grandmaster!"
                    elif logic == 2: remark = "Don't worry, you'll improve with practice. Perhaps try a lower difficulty?"
                    print(f"You lost against the Bot on {diff} difficulty.")
                    pprint(remark)
                    if color == 0: black += 1
                    else: white += 1
                total += 1
            case '3' | 'load':
                default = 'board.txt'
                try: file = input(f"Press enter to load '{default}', or input file name: ").lower()
                except:
                    print("Quitting...")
                    break
                if not file: file = 'board.txt'
                elif file in ['x','cancel']:
                    print("Cancelled.")
                    continue
                board = Board()
                if board.load(file): print("Loading game...")
                else:
                    print(f"Unable to find '{file}'.")
                    continue
                result = board.mainloop()
                if result == 1: white += 1
                elif result == 0: black += 1
                else:
                    white += 1
                    black += 1
                total += 1
            case '4' | 'stats':
                wperc = 0 if white+black == 0 else 100*white/(white+black)
                bperc = 0 if white+black == 0 else 100-wperc
                pprint(f"Total Games: {total} | White wins: {white} ({wperc}%) | Black wins: {black} ({bperc}%) | Wins vs Bot: {botwins}")
            case 'admin':
                ADMIN = not ADMIN
                print(f"ADMIN toggled {'ON' if ADMIN else 'OFF'}.")
            case 'debug':
                DEBUG = not DEBUG
                print(f"DEBUG toggled {'ON' if DEBUG else 'OFF'}.")
            case 'help' | 'h':
                pprint(HELP)
            case '0' | 'quit' | 'x':
                pprint("Quitting...")
                break
            case '':
                if quitcheck: break
                print("Input nothing again to quit.")
                quitcheck = True
            case _:
                print("Unknown input. Type 'help'.")

# ____________________________________________________________________________________________________________________________
pygame.font.init()

WIDTH, HEIGHT = 800,800
GAPX,GAPY = 10, 10
BG = (45, 34, 24)
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
BORDER = True
SQUAREBORDER_COLOR = (45, 34, 24)
SQUAREBORDER_WIDTH = 1
BLACKPIECE = (0,0,0)
WHITEPIECE = (255,255,255)
MOVEOPTION = (255,100,100)
MOVEOPTION_BORDER = 0
CAPTUREOPTION = (255,0,0)
CAPTUREOPTION_BORDER = 0
HIGHLIGHT = (0,255,0)
HIGHLIGHT_BORDER = 0
HIGHLIGHT2 = (100,255,100)
HIGHLIGHT2_BORDER = 0
PROMOCOLOR = (255,215,0)
PROMOINFO = (78,45,0)
INFO = True
INFODARK = LIGHT
INFOLIGHT = DARK
MENUBG = (255,255,255)
MENUFONT = (0,0,0)
MENUBUTTONS = (0,0,0)
ENDFONT = (0,200,255)
ENDBUTTONS = (255,255,255)
ENDBUTTONSFONT = (0,0,0)

WIDTH -= (WIDTH-2*GAPX)%8
HEIGHT -= (HEIGHT-2*GAPY)%8
BOARDW,BOARDH = WIDTH-2*GAPX,HEIGHT-2*GAPY

class PyBoard(Board):
    def __init__(self, win, filled: bool = False):
        super().__init__(filled=filled)
        self.win = win
        self.w, self.h = BOARDW//8,BOARDH//8
        self.piecefont = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", min(self.w,self.h))
        self.infofont = pygame.sysfont.SysFont("arial",min(self.w,self.h)//5)
        self.menufont = pygame.sysfont.SysFont("verdana",min(WIDTH,HEIGHT)//10)
        self.menuoptionfont = pygame.sysfont.SysFont("arial",min(WIDTH,HEIGHT)//15)
        self.font = pygame.sysfont.SysFont("verdana",min(WIDTH,HEIGHT)//30)
        self.swapped = False
    def draw(self, update: bool = False): # Still need to add info (letters/numbers)
        """Renders the full board."""
        self.win.fill(BG)
        for h in range(8):
            for w in range(8):
                self.drawsquare((w,h))
        if update: pygame.display.update()
    def drawmenu(self, stats: tuple | None = None, update: bool = True, cont: tuple | None = None) -> list:
        """Renders initial menu. Stats = (white wins, black wins). Returns list of buttons."""
        buttons = []
        self.win.fill(MENUBG)
        i = self.menufont.render("PyChess", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,HEIGHT//5))
        self.win.blit(i,surf)
        buttonwidth = WIDTH//2
        H = 0 if cont is None else 1
        if stats is not None:
            w, b = stats
            wperc = 0 if w+b == 0 else 100*w/(w+b)
            bperc = 0 if w+b == 0 else 100-wperc
            i = self.font.render(f"White wins: {w} ({wperc:.0f}%) | Black wins: {b} ({bperc:.0f}%)", True, MENUFONT)
            self.win.blit(i,i.get_rect(center=(WIDTH//2,3*HEIGHT//10)))
        i = self.menuoptionfont.render("PLAY", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,(4+H)*HEIGHT//10))
        self.win.blit(i, surf)
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
        buttons.append(surf)
        i = self.menuoptionfont.render("PLAY VS BOT", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,(5+H)*HEIGHT//10))
        self.win.blit(i, surf)
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
        buttons.append(surf)
        i = self.menuoptionfont.render("LOAD", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,(6+H)*HEIGHT//10))
        self.win.blit(i, surf)
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
        buttons.append(surf)
        i = self.menuoptionfont.render("SETTINGS", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,(7+H)*HEIGHT//10))
        self.win.blit(i, surf)
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
        buttons.append(surf)
        i = self.menuoptionfont.render("QUIT", True, MENUFONT)
        surf = i.get_rect(center=(WIDTH//2,(8+H)*HEIGHT//10))
        self.win.blit(i, surf)
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
        buttons.append(surf)
        if cont is not None:
            i = self.menuoptionfont.render("CONTINUE", True, MENUFONT)
            surf = i.get_rect(center=(WIDTH//2,4*HEIGHT//10))
            self.win.blit(i, surf)
            center, surf.width = surf.center, buttonwidth
            surf.center = center
            pygame.draw.rect(self.win, MENUBUTTONS, surf.inflate(0, WIDTH//200), 1)
            buttons.append(surf)
        if update: pygame.display.update()
        return buttons
    def drawend(self, result: int) -> list:
        """Renders screen at end of match. Returns list of buttons."""
        buttons = []
        self.draw()
        w, h = self.win.get_size()
        small = pygame.transform.smoothscale(self.win, (w//3, h//3))
        blurred = pygame.transform.smoothscale(small, (w, h))
        self.win.blit(blurred, (0, 0))
        result1 = "CHECKMATE" if result in [0,1] else "STALEMATE"
        i = self.menufont.render(result1, True, ENDFONT)
        surf = i.get_rect(center=(WIDTH//2,3*HEIGHT//10))
        self.win.blit(i,surf)
        result1 = "White wins!" if result == 1 else "Black wins!" if result == 0 else "It's a draw."
        col = WHITEPIECE if result == 1 else BLACKPIECE
        i = self.menufont.render(result1, True, col)
        surf = i.get_rect(center=(WIDTH//2,4.5*HEIGHT//10))
        self.win.blit(i,surf)
        buttonwidth = WIDTH//2
        i = self.menuoptionfont.render("PLAY AGAIN", True, ENDBUTTONSFONT)
        surf = i.get_rect(center=(WIDTH//2,6*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, ENDBUTTONS, surf.inflate(0, WIDTH//40), 0)
        self.win.blit(i, text)
        buttons.append(surf)
        i = self.menuoptionfont.render("MENU", True, ENDBUTTONSFONT)
        surf = i.get_rect(center=(WIDTH//2,7.5*HEIGHT//10))
        text = surf.copy()
        center, surf.width = surf.center, buttonwidth
        surf.center = center
        pygame.draw.rect(self.win, ENDBUTTONS, surf.inflate(0, WIDTH//40), 0)
        self.win.blit(i, text)
        buttons.append(surf)
        pygame.display.update()
        return buttons
    def drawpromo(self, square: str) -> list:
        """Handles pawn promotion rendering. drawsquare requires promo=(type: str, color: int, text: str)"""
        square = self.ind(square)
        piece = self.lookup(square)
        if piece is None: return []
        c = piece.color
        xcoord = square[0]+1 if square[0] != 7 else 6
        squares = [(xcoord,i) for i in range(4)] if square[1] == 0 else [(xcoord,7-i) for i in range(4)]
        for i,O in enumerate([('queen',c,'1'),('knight',c,'2'),('rook',c,'3'),('bishop',c,'4')]):
            self.drawsquare(squares[i],promo=O)
        return squares
    def drawsquare(self, location: tuple, highlight: bool = False, moveoption: bool = False, piece: bool = True, hl2: bool = False, captureoption: bool = False, promo: tuple | None = None):
        """Draws a single square. Location is (x,y)."""
        w,h = location
        col = DARK if (w+h)%2 else LIGHT
        if promo is not None: col = PROMOCOLOR
        coord = (GAPX+w*self.w,GAPY+h*self.h if self.swapped else GAPY+BOARDH-(1+h)*self.h)
        pygame.draw.rect(self.win,col,(coord[0],coord[1],self.w,self.h))
        if BORDER: pygame.draw.rect(self.win,SQUAREBORDER_COLOR,(coord[0],coord[1],self.w,self.h),SQUAREBORDER_WIDTH)
        if moveoption:
            pygame.draw.rect(self.win,MOVEOPTION,(coord[0],coord[1],self.w,self.h),MOVEOPTION_BORDER)
            if MOVEOPTION_BORDER == 0 and BORDER: pygame.draw.rect(self.win,SQUAREBORDER_COLOR,(coord[0],coord[1],self.w,self.h),SQUAREBORDER_WIDTH)
        if captureoption:
            pygame.draw.rect(self.win,CAPTUREOPTION,(coord[0],coord[1],self.w,self.h),CAPTUREOPTION_BORDER)
            if CAPTUREOPTION_BORDER == 0 and BORDER: pygame.draw.rect(self.win,SQUAREBORDER_COLOR,(coord[0],coord[1],self.w,self.h),SQUAREBORDER_WIDTH)
        if highlight:
            pygame.draw.rect(self.win,HIGHLIGHT,(coord[0],coord[1],self.w,self.h),HIGHLIGHT_BORDER)
            if HIGHLIGHT_BORDER == 0 and BORDER: pygame.draw.rect(self.win,SQUAREBORDER_COLOR,(coord[0],coord[1],self.w,self.h),SQUAREBORDER_WIDTH)
        if hl2:
            pygame.draw.rect(self.win,HIGHLIGHT2,(coord[0],coord[1],self.w,self.h),HIGHLIGHT2_BORDER)
            if HIGHLIGHT2_BORDER == 0 and BORDER: pygame.draw.rect(self.win,SQUAREBORDER_COLOR,(coord[0],coord[1],self.w,self.h),SQUAREBORDER_WIDTH)
        p = self[(w,h)] if promo is None else Piece(promo[0],promo[1])
        if p is not None and piece:
            p = self.piecefont.render(p.symbol, True, WHITEPIECE if p.color else BLACKPIECE)
            center = p.get_rect(center=(coord[0]+self.w//2,coord[1]+self.h//2-SQUAREBORDER_WIDTH))
            self.win.blit(p,center)
        if promo is not None:
            i = self.infofont.render(promo[2], True, PROMOINFO)
            center = i.get_rect(center=(coord[0]+self.w//10,coord[1]+3*self.h//20))
            self.win.blit(i,center)
            return
        if w == 0 and INFO:
            nums = '12345678'
            if self.swapped: nums = nums[::-1]
            i = self.infofont.render(nums[h], True, INFODARK if col == DARK else INFOLIGHT)
            center = i.get_rect(center=(coord[0]+self.w//10,coord[1]+3*self.h//20))
            self.win.blit(i,center)
        if h == 0 and INFO:
            letters = 'abcdefgh'
            if self.swapped: letters = letters[::-1]
            i = self.infofont.render(letters[w], True, INFODARK if col == DARK else INFOLIGHT)
            center = i.get_rect(center=(coord[0]+9*self.w//10,coord[1]+17*self.h//20))
            self.win.blit(i,center)

    def drawoptions(self, square: tuple, options: list, captureoptions: list) -> None:
        p = self.lookup(square)
        if p is None or options is None: return []
        triggers = []
        c,lc = None,None
        if 'castle' in options:
            c = (6,0) if p.color == 1 else (6,7)
            self.drawsquare(c,moveoption=True)
        if 'longcastle' in options:
            lc = (1,0) if p.color == 1 else (1,7)
            self.drawsquare(lc,moveoption=True)
        if c is not None: triggers.append(c)
        if lc is not None: triggers.append(lc)
        for o in options:
            if type(o) == tuple:
                triggers.append(o)
                self.drawsquare(o,moveoption=True)
        for o in captureoptions:
            self.drawsquare(o,captureoption=True)
        return triggers+captureoptions
    def getsquare(self, pos: tuple) -> tuple | None:
        """Returns the square index from the mouse position, or None."""
        X,Y = pos
        x,y = (X-GAPX)//self.w, (Y-GAPY)//self.h if self.swapped else -1-((Y-GAPY-BOARDH)//self.h)
        return (x,y) if 0<=x<8 and 0<=y<8 else None

def pyloop():
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PyChess")
    board = PyBoard(window)
    running = True
    color = 1
    MOVE = None
    SWAPPING = False
    COLOROVERRIDE = False
    LOCKTURN = False
    white, black, total, botwins = 0, 0, 0, 0
    menu = True
    settings, match, endscreen = False, False, False
    result = -1
    save = None # (color)
    while running:
        if menu: buttons = board.drawmenu((white,black),cont = save)
        while menu:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if LOGS: pprint("Quitting...")
                    pygame.quit()
                    menu, running = False, False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
                    if buttons[0].collidepoint(mx,my): # Play
                        color = 1
                        MOVE = None
                        save = None
                        result = -1
                        board.setup()
                        menu, match = False, True
                    elif buttons[1].collidepoint(mx,my): # Play vs bot
                        print("PLAY VS BOT")
                    elif buttons[2].collidepoint(mx,my): # Load board
                        print("LOAD")
                    elif buttons[3].collidepoint(mx,my): # Settings
                        print("SETTINGS")
                    elif buttons[4].collidepoint(mx,my): # Quit
                        if LOGS: pprint("Quitting...")
                        menu, running = False, False
                        pygame.quit()
                    elif save is not None and buttons[5].collidepoint(mx,my): # Continue
                        color = save[0]
                        MOVE = None
                        menu, match = False, True
        while settings:
            pass
        while match:
            turn = True
            highlighted = None
            dragging = False
            lastselect = (-1,-1)
            options = []
            captureoptions = []
            optionsTriggers = []
            outcome = False
            board.draw(update=True)
            if save is not None and len(save)>1 and isinstance(save[1],str):
                turn = False
                outcome = save[1]
                save = None
            while turn:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if LOGS: pprint("Quitting...")
                        pygame.quit()
                        turn, match, running = False, False, False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_x]:
                            turn, match = False, False
                            menu = True
                            save = (color,)
                            break
                    elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]: # Left click
                        pos = pygame.mouse.get_pos()
                        square = board.getsquare(pos)
                        board.draw()
                        if square in optionsTriggers: # Handleturn
                            turn = False
                            if 'castle' in options and square in [(6,0),(6,7)]: MOVE = 'castle'
                            elif 'longcastle' in options and square in [(1,0),(1,7)]: MOVE = 'longcastle'
                            else: MOVE = f"{board.code(highlighted)} {board.code(square)}"
                            break
                        if square is not None:
                            if square == highlighted: lastselect = square
                            if highlighted is not None: 
                                board.drawsquare(square, highlight=highlighted)
                                options, captureoptions, optionsTriggers = [],[],[]
                            highlighted = square
                            board.drawsquare(square,highlight=True)
                            p = board.lookup(square)
                            if p is not None and (p.color == color or COLOROVERRIDE):
                                dragging = True
                                options = board.move_options(square,CHECK=True,CASTLING=True)
                                captureoptions = board.capture_options(square,CHECK=True)
                            optionsTriggers = board.drawoptions(square,options, captureoptions)
                        pygame.display.update()
                    elif event.type == pygame.MOUSEBUTTONUP:
                        pos = pygame.mouse.get_pos()
                        square = board.getsquare(pos)
                        if lastselect == square:
                            lastselect = (-1,-1)
                            highlighted = None
                            options,captureoptions,optionsTriggers = [],[],[]
                            board.draw(update=True)
                        if dragging:
                            dragging = False
                            if square in optionsTriggers: # Handleturn
                                turn = False
                                if 'castle' in options and square in [(6,0),(6,7)]: MOVE = 'castle'
                                elif 'longcastle' in options and square in [(1,0),(1,7)]: MOVE = 'longcastle'
                                else: MOVE = f"{board.code(highlighted)} {board.code(square)}"
                                break
                            board.draw()
                            if highlighted is not None:
                                board.drawsquare(location=highlighted,highlight=True)
                                board.drawoptions(highlighted,options,captureoptions)
                            pygame.display.update()
                    elif pygame.mouse.get_pressed()[0] and highlighted is not None and board[highlighted] is not None and dragging:
                        pos = pygame.mouse.get_pos()
                        currsquare = board.getsquare(pos)
                        board.draw()
                        if square is not None:
                            board.drawsquare(square,highlight=True,piece=False)
                            board.drawoptions(square,options,captureoptions)
                        if currsquare is not None: 
                            if currsquare == square: board.drawsquare(currsquare,hl2=True, piece=False)
                            else: board.drawsquare(currsquare,hl2=True)
                        p = board[highlighted]
                        p = board.piecefont.render(p.symbol, True, WHITEPIECE if p.color else BLACKPIECE)
                        center = p.get_rect(center=(pos[0],pos[1]))
                        board.win.blit(p,center)
                        pygame.display.update()
            if not match: break
            if COLOROVERRIDE and not outcome:
                outcome = board.handle_turn(MOVE, color)
                if not outcome: outcome = board.handle_turn(MOVE, int(not color))
            elif not outcome:
                outcome = board.handle_turn(MOVE, color)
            if isinstance(outcome, str): # Pawn promotion
                board.draw()
                squares = board.drawpromo(square)
                pygame.display.update()
                prompting = True
                if LOGS: print(f"{"White" if color == 1 else "Black"} promoting pawn on {outcome}.\n(Q/Space = Queen | N = Knight | R = Rook | B = Bishop)")
                while prompting:
                    promo = None
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            if LOGS: pprint("Quitting...")
                            pygame.quit()
                            prompting, match = False, False
                        elif event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_q,pygame.K_1,pygame.K_SPACE]:
                                promo = 'queen'
                                prompting = False
                            elif event.key in [pygame.K_n,pygame.K_2]:
                                promo = 'knight'
                                prompting = False
                            elif event.key in [pygame.K_r,pygame.K_3]:
                                promo = 'rook'
                                prompting = False
                            elif event.key in [pygame.K_b,pygame.K_4]:
                                promo = 'bishop'
                                prompting = False
                            elif event.key in [pygame.K_x, pygame.K_ESCAPE]:
                                turn, match, prompting = False, False, False
                                menu = True
                                save = (color,outcome)
                                break
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            pos = pygame.mouse.get_pos()
                            square = board.getsquare(pos)
                            if square in squares:
                                promo = ['queen','knight','rook','bishop'][squares.index(square)]
                                prompting = False
                    if not prompting: break
                if not match: break
                if promo is not None:
                    board.setpiece(outcome, name=promo)
                    if LOGS: print(f"{"White" if color == 1 else "Black"} promoted pawn on {outcome} to a {promo.capitalize()}.")
            if not LOCKTURN: 
                color = int(not color)
                if SWAPPING: board.swapped = not board.swapped
            if board.stalemate(color) or board.drawmate():
                if LOGS: print(f"[bold]Stalemate![/bold]")
                result = -1
                total += 1
            if board.checkmate(color):
                if LOGS: print(f"\n[bold]Checkmate![/bold] {f"[{WHITE}]White[/{WHITE}]" if color==0 else f"[{BLACK}]Black[/{BLACK}]"} has won.")
                result = int(not color)
                total += 1
                if result == 0: black += 1
                else: white += 1
                match, menu = False, True
                endscreen = True
        if endscreen: buttons = board.drawend(result)
        while endscreen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if LOGS: pprint("Quitting...")
                    pygame.quit()
                    endscreen, running = False, False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        endscreen, menu = False, True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
                    if buttons[0].collidepoint(mx,my): # Play again
                        endscreen, menu, match = False, False, True
                    elif buttons[1].collidepoint(mx,my): # Menu
                        endscreen, menu = False, True

#TODO: Change settings to keybind help (?), admin options (move opponent pieces, set up board), finish initial screen, turn indicator, timer, custom pieces...

if __name__ == '__main__':
    # superloop()
    pyloop()
