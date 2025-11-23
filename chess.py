
#TODO: Actually make it work, swap board vertically for black's turn, make in pygame?

class Board:
    def __init__(self, N: tuple = (8,8), filled: bool = False, type: str | None = 'Chess'):
        """Chessboard."""
        self.W, self.H = N
        self.type = type
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
        self.symbol_dict = {"king":"♔","queen":"♕","rook":"♖","bishop":"♗","knight":"♘","pawn":"♙"}
        self.simplified_dict = {"king":"k","queen":"q","rook":"r","bishop":"b","knight":"n","pawn":"p"}
        self.gap = 2 # board str
        self.blank = '▢'
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
        for i,line in enumerate(self.board):
            res.append("|")
            res[i] += ''.join([p.simplified.center(self.gap) if p is not None else "".center(self.gap) for p in line])+"|"
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
        self.bord = [[None for _ in range(self.W)] for _ in range(self.H)]
    def move(self,square1: str | tuple, square2: str | tuple) -> bool:
        """Move piece to chosen square. Requires square to be empty."""
        (x1,y1),(x2,y2) = self.ind(square1),self.ind(square2)
        if self.lookup((x2,y2)) is None:
            self.board[y2][x2] = self.board[y1][x1]
            self.board[y1][x1] = None
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
            return True
        else: return False
    def move_options(self, square: str | tuple) -> list:
        """Returns a list of all possible move options for square."""
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        return piece.move((x,y))
    def convert(self, values: str | tuple | list) -> str | tuple | list:
        """Converts index to notation and viceversa."""
        if type(values) != list: return self.code(values) if type(values)==tuple else self.ind(values)
        return [self.code(v) if type(v)==tuple else self.ind(v) for v in values]
    def capture_options(self, square: str | tuple) -> list:
        x,y = self.ind(square)
        piece = self.lookup((x,y))
        if piece is None: return []
        return piece.capture((x,y))
    def advantage(self, color: int = 1) -> int:
        """Returns point advantage for color."""
        if color == 0: color = -1
        return color*(sum(p.value() for p in self.black_captured)-sum(p.value() for p in self.white_captured))
    def turn(self, color: int = 1, info: bool = True) -> int:
        """Turn of specified color."""
        while True:
            move = input("> ")
            if move in ['h','help','format']:
                print("""Use chess notation or x,y pairs.
- A = Shows all available moves from square. Ex: a1
- A B = Moves piece in square to specified. Ex: 1,2 1,4""")
                continue
            elif move in ['stop','cancel','x','quit']: 
                print("Cancelled.")
                break
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
                    print("Move options:",self.convert(self.move_options(move)))
                    print("\nCapture options:",self.convert(self.capture_options(move)))
                    continue
                elif len(move) == 2:
                    moveA,moveB = move
                    if ',' in moveA: 
                        x,y = moveA.split(',')
                        moveA = (int(x),int(y))
                    if ',' in moveB: 
                        x,y = moveB.split(',')
                        moveB = (int(x),int(y))
                    opt, cap = self.move_options(moveA), self.capture_options(moveA)
                    if self.ind(moveB) in opt:
                        self.move(moveA,moveB)
                        if info: print(f"{self.lookup(moveB).type.capitalize()} moved to {self.code(moveB)}.")
                        break
                    elif self.ind(moveB) in cap:
                        capt = self.lookup(moveB).type.capitalize()
                        self.capture(moveA, moveB)
                        if info: print(f"{self.lookup(moveB).type.capitalize()} captured {capt} on {self.code(moveB)}.")
                    else:
                        print("Invalid move. Try typing the piece to see available moves.")
                else: print("Unknown input. Type 'help' for info.")
            except:
                print("Unknown input. Type 'help' for info.")
        print('\n'+str(self))
        if info: print(f"Material advantage: {self.advantage(color=color)}")

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
    def capture(self, location: tuple) -> list:
        """Returns possible capture options as list of (x,y)."""
        x,y = location
        options = []
        match self.type:
            case 'pawn':
                if self.color==1: options += [(x+1,y+1),(x-1,y+1)]
                elif self.color==0: options += [(x+1,y-1),(x-1,y-1)]
            case _:
                options = self.options(location)
        return list(filter(lambda x: (P:=self.board.lookup(x)) is not None and P.color != self.color,options))

if __name__ == '__main__':
    b = Board(filled=True)
    print(b)
