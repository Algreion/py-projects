def print_board(board):
    for row in board:
        print("","-"*13)
        print(" |"," | ".join(row),"|")
    print("","-"*13)

def winner(board, player):
    for row in board:
        if all(s == player for s in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False

def play(player="X", x_score = 0, o_score = 0, counting = False):
    """Position either through two numbers (row column) or position 1-9"""
    board = [[" " for _ in range(3)] for _ in range(3)]
    turn = 1
    while True:
        print_board(board)
        if turn > 9: break
        try:
            move = input(f"Player {player}, enter your move: ")
            if move == "" or move == " ": 
                print("Game cancelled.")
                if counting: return (x_score, o_score)
                return
            move = move.split()
            if any(not i.isdigit() for i in move):
                print("Invalid move, only numbers allowed.\n")
                continue
        except (Exception,KeyboardInterrupt,EOFError):
            print("Game cancelled.")
            if counting: return (x_score, o_score)
            return 
        if len(move) == 2:
            row, col = map(int, move)
            row, col = row-1, col-1
        elif len(move) == 1:
            move = int(move[0]) - 1
            row = move // 3
            col = move % 3
        else:
            print("Invalid move, input either 1 or 2 numbers.\n")
            continue
        if not (0 <= col < 3 and 0 <= row < 3):
            print("Invalid move, number is too high/low.\n")
            continue
        if board[row][col] == " ":
            board[row][col] = player
            if winner(board, player):
                print_board(board)
                print(f"Player {player} wins!")
                if player == "O": o_score += 1
                else: x_score += 1
                if counting: return (x_score, o_score)
                return
            player = "O" if player == "X" else "X"
            turn += 1
        else:
            print("Invalid move, that spot is occupied.\n")
    print("It's a tie.")
    if counting: return (x_score, o_score)

if __name__ == "__main__":
    ask = True
    x, o = 0, 0
    while ask:
        pair = play("X", o, x, True)
        x, o = pair[0], pair[1]
        print(f"X wins: {x}")
        print(f"O wins: {o}")
        ask = input("Play again (y/n)? ")
        if ask != "y": ask = False
    print("Thanks for playing!")
    
