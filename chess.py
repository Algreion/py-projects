class Board:
    def __init__(self, N: tuple = (8,8), filled: bool = False, type: str | None = 'Chess'):
        """Chessboard."""
        self.W, self.H = N
        self.type = type
        self.board = [[None for _ in range(self.W)] for _ in range(self.H)]
        self.symbol_dict = {"king":"♔","queen":"♕","rook":"♖","bishop":"♗","knight":"♘","pawn":"♙"}
        self.simplified_dict = {"king":"k","queen":"q","rook":"r","bishop":"b","knight":"n","pawn":"p"}
        self.gap = 2 # board str
        self.white_captured,self.black_captured = set(),set()
        self.move_dict = {
        "king":    [(1,0), (-1,0), (0,1), (0,-1),
                    (1,1), (1,-1), (-1,1), (-1,-1)],

        "queen":   [(1,0), (-1,0), (0,1), (0,-1),
                    (1,1), (1,-1), (-1,1), (-1,-1)],

        "rook":    [(1,0), (-1,0), (0,1), (0,-1)],

        "bishop":  [(1,1), (1,-1), (-1,1), (-1,-1)],

        "knight":  [(2,1), (2,-1), (-2,1), (-2,-1),
                    (1,2), (1,-2), (-1,2), (-1,-2)],

        "pawn":    [(0,1)]
        } # unit vectors for sliding pieces
        self.points = {'pawn':1,'knight':3,'bishop':3,'rook':5,'queen':9}
        if filled: self.setup()
    def __repr__(self):
        return f"Board(width={self.W},height={self.H},type={self.type})"
    def __str__(self):
        res = []
        for i,line in enumerate(self.board[::-1]):
            res.append("|")
            res[i] += ''.join([str(p).center(self.gap) if p is not None else "".center(self.gap) for p in line])+"|"
        return "\n".join(res)
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
        
    def advantage(self, color: int = 1) -> int:
        """Returns point advantage for color."""
        if color == 0: color = -1
        return color*(sum(p.value() for p in self.black_captured)-sum(p.value() for p in self.white_captured))

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
    def move(self) -> list:
        """Returns possible move options."""
        match self.type:
            case 'pawn': pass
            case 'knight': pass
            case 'rook': pass
            case 'bishop': pass
            case 'queen': pass
            case 'king': pass
    def capture(self) -> list:
        """Returns possible capture options."""
        match self.type:
            case 'pawn': pass
            case 'knight': pass
            case 'rook': pass
            case 'bishop': pass
            case 'queen': pass
            case 'king': pass

b = Board(filled=True)
print(b)
