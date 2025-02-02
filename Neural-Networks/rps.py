from collections import deque
from micrograd import *

def rps():
    ACTIVATION = 7
    print("r = Rock, p = Paper, s = Scissors")
    lookup = {'r': -1, 'p': 0, 's': 1}
    win = {-1: 0, 0: 1, 1: -1}
    ilookup = {-1: 'Rock', 0: 'Paper', 1: 'Scissors'}
    n = 0
    counts = {-1: 0, 0: 0, 1: 0}
    probability = [0,0,0]
    nn = MLP(ACTIVATION + 3,[8,8,1])
    player_score = 0
    ai_score = 0
    pattern = deque()
    try:
        while (x:=input("> ")):
            try:
                x = lookup[x]
            except:
                print("Invalid command. Type 'r', 'p' or 's'.")
                continue
            n += 1
            counts[x] += 1
            if n > ACTIVATION:
                probability = [counts[-1]/n,counts[0]/n,counts[1]/n]
                pattern.popleft()
                pattern.append(x)
                inputs = list(pattern)+probability
                expected = x
                res = win[round(train(nn, inputs, expected).data)]
            else:
                pattern.append(x)
                res = random.randint(-1,1)
            print("AI:",ilookup[res],end=" | ")
            if win[res] == x:
                print("Won!")
                player_score += 1
            elif res == x:
                print("Tie!")
            else:
                print("Lost!")
                ai_score += 1
            print(f"Score: P | {player_score} - AI | {ai_score}")
    finally:
        print(f"Thanks for playing! Final score: {player_score}, {ai_score}")

if __name__ == "__main__":
    rps()
