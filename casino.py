import random
from collections import deque
from rich import print,text

#TODO | UNO, add cheating options with skill, finish casino mainloop, add chess, add roulette

N2W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten"}
W2N = dict([(w,n) for n,w in N2W.items()])
FACEDOWN = "purple"
RICH = True
SEP = "________________________________"
BLACKJACK_HELP = """
1 / H - Hit
2 / S - Stand
3 / D - Double down (x2 bet, take 1 card, stand)"""
CASINO_HELP = """
1 / P - Play a game
2 / S - View current status
3 / C - Manage characters
4 / T - Train skill or earn money
5 / Z - Cheats
6 / X - Quit"""
SKILLCAP = 100
DEALERSKILL = 50
SKILLRATE = 0.1 # Skill gain multiplier

class Player:
    def __init__(self, name: str= "Player", money: float = 100, skill: int = 0, age: int = 18):
        self.name = name
        self.age = age
        self.money = money
        self.skill = skill
        self.wins = 0
        self.losses = 0
    def __repr__(self):
        return f"Player(name={self.name},money={self.money},skill={self.skill},age={self.age})"
    def __str__(self):
        return self.name

class Card:
    def __init__(self, name: str, suit: str):
        self.name = name.lower()
        self.suit = suit.lower()
        self.color = "white" if suit in ["spades","clubs"] else "red"
        self.visible = True
        self.marked = False
    def __repr__(self):
        return f"Card(name={self.name},suit={self.suit}{",visible" if self.visible else ""}{",marked" if self.marked else ""})"
    def __str__(self):
        d = {"ace":'A',"spades": '♠',"clubs":'♣',"hearts":'♥',"diamonds":'♦',"jack":'J',"queen":'Q',"king":'K'}
        try:
            return f"{d[self.name] if self.name in d else W2N[self.name]}{d[self.suit]}" if self.visible else "?"
        except:
            return "?"
    def dyncolor(self):
        if self.visible: return self.color
        else: return FACEDOWN
    def __rich__(self):
        return text.Text.from_markup(f"[{self.dyncolor()}]"+str(self)+f"[/{self.dyncolor()}]")

class UnoCard:
    def __init__(self, name: str, color: str):
        self.name = str(name).lower()
        self.type = color.lower()
        self.color = {"red":"bright_red","blue":"dodger_blue2","green":"green3","yellow":"yellow1","black":"white"}[self.type]
        self.visible = False
        self.marked = False
    def __repr__(self):
        return f"UnoCard(name={self.name},color={self.color}){",visible" if self.visible else ""}{",marked" if self.marked else ""}"
    def __str__(self):
        d = {"skip":'⦸',"reverse":'↻',"draw2":'+2',"wild":'W',"draw4":'+4'}
        return f"{d[self.name] if self.name in d else self.name}"
    def __rich__(self):
        x = str(self)
        return text.Text.from_markup(f"[{self.color}]"+x+f"[/{self.color}]")

class Deck:
    def __init__(self, type: str | None | int = "classic"):
        self.deck = deque()
        self.size = 0
        self.type = None
        if type is not None:
            match str(type).lower():
                case "uno" | "1":
                    self.type = "UNO"
                    self.init(1)
                case "empty" | "-1":
                    self.init(-1)
                case _:
                    self.type = "Classic"
                    self.init(0)
    def __repr__(self):
        return f"Deck(cards={self.size},type={self.type})"
    def __str__(self):
        return str([str(card) for card in self.deck]).replace("'",'')
    def __rich__(self):
        return text.Text.from_markup(str(([f"[{card.dyncolor()}]"+str(card)+f"[/{card.dyncolor()}]" for card in self.deck])).replace("'",''))
    def __getitem__(self, i: int):
        return self.deck[i]
    def __setitem__(self, i, c):
        self.deck[i] = c
    def __delitem__(self, i):
        del self.deck[i]
    def shuffle(self):
        random.shuffle(self.deck)
    def draw(self, n: int = 1) -> list:
        res,n = [],min(n,self.size)
        for _ in range(n):
            res.append(self.deck.popleft())
            self.size -= 1
        return res
    def peek(self, n: int = 1) -> list:
        res,n = [],min(n,self.size)
        for i in range(n):
            res.append(self.deck[i])
        return res
    def insert(self, cards: Card | list, bottom = True) -> None:
        if isinstance(cards,Card): cards = [cards]
        for card in cards:
            if bottom:
                self.deck.append(card)
            else:
                self.deck.appendleft(card)
            self.size += 1
    def init(self, type: int = 0) -> None:
        """Initialize cards. 
        0 = Standard 52 cards | 1 = UNO
        -1 = Empty deck"""
        if type == 0:
            self.deck = deque()
            for suit in ["spades","hearts","clubs","diamonds"]:
                for card in ["ace","two","three","four","five","six","seven","eight","nine","ten","jack","queen","king"]:
                    self.deck.append(Card(card,suit))
        elif type == 1:
            self.deck = deque()
            for color in ["red","blue","green","yellow"]:
                for card in [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,"skip","skip","reverse","reverse","draw2","draw2"]:
                    self.deck.append(UnoCard(card,color))
            for card in ["wild" for _ in range(4)]+["draw4" for _ in range(4)]:
                self.deck.append(UnoCard(card,"black"))
        elif type == -1:
            self.deck = deque()
        self.size = len(self.deck)

class Hand:
    def __init__(self, player: Player | None = None, game: str | None = None):
        self.player = player
        self.game = game
        self.size = 0
        self.hand = []
    def __getitem__(self, i: int):
        return self.hand[i]
    def __setitem__(self, i, c):
        self.hand[i] = c
    def __delitem__(self, i):
        del self.hand[i]
    def __len__(self):
        return len(self.hand)
    def __repr__(self):
        return f"Hand(cards={len(self.hand)}{f",player={self.player}" if self.player is not None else ""}{f",game={self.game}" if self.game is not None else ""})"
    def __str__(self):
        return str([str(card) for card in self.hand]).replace("'",'')
    def __rich__(self):
        return text.Text.from_markup(str(([f"[{card.dyncolor()}]"+str(card)+f"[/{card.dyncolor()}]" for card in self.hand])).replace("'",''))
    def draw(self, deck: Deck, n: int = 1) -> None:
        self.hand.extend(deck.draw(n))
        self.size = len(self)
    def discard(self, deck: Deck | None = None, card: Card | int | None = None, bottom: bool = True):
        """Put a card back in the deck (or throws away if deck=None). 
        If card=None, discards all."""
        if card is None:
            if deck is not None: deck.insert(self.hand,bottom)
            self.hand = []
        else:
            if isinstance(card,Card) and card in self.hand: card = self.hand.index(card)
            if deck is not None: deck.insert(self.hand.pop(card))
            else: self.hand.pop(card)
        self.size = len(self)
    def add(self, card: Card):
        """Adds a card to the hand."""
        self.hand.append(card)
        self.size = len(self)
Player()
class Casino:
    def __init__(self, name: str = "Casino", money: float = 1e6, dealer: Player | None = None):
        self.name = name
        self.money = money
        self.size = 0
        self.ID = 0
        self.db = dict()
        self.dealer = dealer
        self.games = {1: Blackjack(casino=self), 2: UNO()}
    def __repr__(self):
        return f"Casino(name={self.name},money={self.money},players={self.size})"
    def __str__(self):
        return self.name
    def create_char(self) -> Player:
        x = input("Name Age\n> ").split()
        match len(x):
            case 0:
                name, age = "Player", 18
            case 1:
                name, age = x[0], 18
            case 2:
                name, age = x[0],int(x[1])
        return Player(name=name,age=age)
    def choose_char(self, active) -> Player | None:
        char = None
        if len(active) > 1:
            print("Choose a character. ")
            x = input(" | ".join(f"{self.playerID(p)} - {p.name}" for p in active),'\n> ')
            if x.isdigit() and int(x) in self.db:
                char = self.db[int(x)]
            else:
                print("Invalid character. Cancelled operation.")
        else: char = self.db[list(active)[0]]
        return char
    def mainloop(self):
        """Full game experience."""
        active = set()
        main = -1
        cheat = False
        print(f"Welcome to [bold {FACEDOWN}]{self.name}[/bold {FACEDOWN}]!")
        print("Your options are as follows. Type 'help' to view them again:",CASINO_HELP)
        while True:
            print(SEP)
            if not active:
                print("You have not yet created a character. Choose the following all in a single line, or leave blank to pick default: ")
                char = self.create_char()
                self.register(char)
                i = self.playerID(char)
                active,main = {i}, i
                print(f"{self.db[i]} registered.")
                continue
            x = input(f"[{self.db[main].name}] > ").lower()
            if x in ['1','p','play']:
                print(f"Choose a game to play. Options:\n{' | '.join(f"{k} - {str(v)}" for k,v in self.games.items())}")
                y = input("> ")
                if not y.isdigit():
                    print("Invalid option.")
                else:
                    self.play(int(y),[self.db[i] for i in active])
            elif x in ['2','s','status']:
                char = self.choose_char(active)
                print(f"Name: {char.name} | Age: {char.age} | Money: {self.money_color(char.money)} | Skill: {char.skill}")
                print(f"Wins: {char.wins} | Losses: {char.losses} | Win rate: {100*char.wins/max(1,(char.wins+char.losses))}%")
            elif x in ['3','c','chars','characters']:
                pass
            elif x in ['4','t','train']:
                pass
            elif x in ['5','z','cheats','cheat']:
                y = input("Cheat options:\n1. Change money | 2. Change skill | 3. Cancel debt\n> ")
                match y:
                    case '1':
                        char = self.choose_char(active)
                        if char is None: continue
                        print(f"Character {char.name} has {self.money_color(char.money)}.")
                        y = input("Input a new cash amount.\n>")
                        try: y = float(y)
                        except:
                            print("Cancelled operation.")
                            continue
                        cheat = True
                        char.money = y
                        print(f"Set {char.name}'s money to {self.money_color(y)}.")
                    case '2':
                        char = self.choose_char(active)
                        if char is None: continue
                        print(f"Character {char.name} has {char.skill} skill.")
                        y = input(f"Input a new skill value. {SKILLCAP} is the maximum.\n> ")
                        try: y = float(y)
                        except:
                            print("Cancelled operation.")
                            continue
                        cheat = True
                        char.skill = min(y,SKILLCAP)
                        print(f"Set {char.name}'s skill to {char.skill}.")
                    case '3':
                        print("Your debt has been forgiven.")
                        for i in active: self.db[i].money = max(self.db[i].money,0)
                        cheat = True
                    case _:
                        print("Invalid option.")
            elif x in ['6','q','quit','x']:
                print(f"Thank you for playing, {self.db[main]}! You made a total of {self.money_color(sum([p.money for p in [self.db[i] for i in active]])-100*len(active))}.")
                quit()
            elif x in ['h','help','format']:
                print(CASINO_HELP)
            else:
                print("Invalid option. Type 'help' to see all options.")
    def play(self, game: int = 1, players: Player | list | None = None):
        """Start up a specific game."""
        if game not in self.games:
            print("Game not available.")
        else:
            gains = []
            if players is not None:
                if isinstance(players,Player): players = [players]
                gains = [p.money for p in players]
            rounds = self.games[game].mainloop(players=players)
            if players is not None and rounds>1:
                gains = [p.money - gains[i] for i,p in enumerate(players)]
                print(f"Played {rounds} rounds.")
                for i in range(len(players)):
                    color = "red1" if gains[i]<0 else "green1" if gains[i]>0 else "white"
                    print(f"{players[i]} {"earned" if color=="green1" else "lost" if color=="red1" else "made"} [{color}]{abs(gains[i])}[/{color}]$.")
    def money_color(self, amount: float) -> str:
        """Returns the color coded amount."""
        POS, NEG, NEUTRAL = "red1","green1","white"
        color = "red1" if amount<0 else "green1" if amount>0 else "white"
        return f"[{color}]{amount}$[/{color}]"
    def players(self):
        return list(self.db.values())
    def playerID(self, player: Player) -> int | None:
        try: return dict([(v,k) for k,v in self.db.items()])[player]
        except: return None
    def register(self, players: Player | list):
        if isinstance(players,Player): players = [players]
        for p in players:
            if self.playerID(p) is not None: continue
            self.db[self.ID] = p
            self.ID += 1
            self.size += 1
    def remove(self, ids: Player | list):
        if isinstance(ids,int): ids = [ids]
        for i in ids:
            if i in self.db:
                del self.db[i]
                self.size -= 1

class Blackjack:
    def __init__(self, players: Player | list | None = None, casino: Casino | None = None):
        """States: BETTING | DEAL | PLAYER_TURN | DEALER_TURN | PAYOUT"""
        self.deck = Deck("classic")
        self.casino = casino
        self.dealer = casino.dealer if casino is not None else Player("Dealer",skill=DEALERSKILL)
        self.dealerhand = Hand(self.dealer,"Blackjack")
        self.rounds = 1
        self.skillgains = [1,0.5,0.25] # Win, Tie, Loss skill increase.
        self.init_players(players)
        self.deck.shuffle()
    def __repr__(self):
        return f"Blackjack(players={len(self.players)},rounds={self.rounds},dealer={self.dealer.name})"
    def __str__(self):
        return "Blackjack"
    def init_players(self, players: Player | list | None = None):
        self.players = None
        if players is None:
            self.players = [Player()]
        else:
            if isinstance(players,Player): players = [players]
            self.players = players
        self.playerhands = dict([(player,Hand(player,"Blackjack")) for player in self.players])
        self.bets = dict([(player,0) for player in self.players])
    def reset(self) -> None:
        """Resets the round, returning all cards to the deck."""
        self.dealerhand.discard(self.deck)
        for h in self.playerhands.values(): h.discard(self.deck)
        self.bets = dict([(player,0) for player in self.players])
        self.deck.shuffle()
        for i in range(self.deck.size):
            self.deck[i].visible = True
    def deal(self) -> bool:
        """Deals two cards to all players."""
        if self.deck.size < 2+2*len(self.players): return False
        self.dealerhand.draw(self.deck,2)
        self.dealerhand[-1].visible = False
        for h in self.playerhands.values(): h.draw(self.deck,2)
        return True
    def draw(self, player: Player, n: int = 1) -> bool:
        """Draws cards in the player's hand."""
        if self.deck.size < n: return False
        if player is self.dealer:
            self.dealerhand.draw(self.deck,n)
        else:
            self.playerhands[player].draw(self.deck,n)
        return True
    def value(self, hand: Hand) -> int:
        """Calculates the hand value."""
        points = 0
        aces = 0
        for c in hand:
            if c.name in W2N:
                points += W2N[c.name]
            elif c.name == "ace": aces += 1
            else: points += 10
        for _ in range(aces):
            if points+11 > 21: points+= 1
            else: points += 11
        return points
    def dealer_turn(self):
        """Handles dealer logic."""
        for c in self.dealerhand: c.visible = True
        while self.value(self.dealerhand) < 17:
            self.draw(self.dealer)
    def payout(self, info: bool = True):
        """Pays players and updates their stats."""
        for p in self.players:
            x = self.determine(self.playerhands[p], info=p.name)
            p.money += x*self.bets[p]
            p.skill += SKILLRATE*self.skillgains[0 if x>0 else 2 if x<0 else 1]
            if info and x!=0: print(f"{p.name} {"earns" if int(x>0) else "pays"} {abs(x*self.bets[p])}$!")
            if self.casino is not None: self.casino.money -= x*self.bets[p]
            p.wins += int(x>0)
            p.losses += int(x<0)
            p.money,p.skill = round(p.money,2),round(p.skill,2)
    def determine(self, hand: Hand, info: str | None = None) -> float:
        """Returns the payout and win/loss info for specific player."""
        value = self.value(hand)
        dealer = self.value(self.dealerhand)
        if len(hand) == 2 and value == 21:
            if info: print(f"[bold green1]{info} got a Natural Blackjack![/bold green1]")
            return 1.5
        elif value > 21:
            if info: print(f"[red1]{info} busts.[/red1]")
            return -1
        elif dealer > 21:
            if info: print(f"[green1]Dealer busts against {info}.[/green1]")
            return 1
        elif value > dealer:
            if info: print(f"[green1]{info} wins.[/green1]")
            return 1
        elif value < dealer:
            if info: print(f"[red1]Dealer wins against {info}.[/red1]")
            return -1
        else:
            if info:  print(f"[cyan]{info} tied.[/cyan]")
            return 0
    def player_turn(self, player: Player):
        """Handles an individual player's turn."""
        print(f"| [bold]{player.name}[/bold]'s turn. |\n\nDealer's hand:")
        print(self.dealerhand)
        while True:
            print(f"\n{player.name}'s hand:")
            print(self.playerhands[player])
            v = self.value(self.playerhands[player])
            print(f"Value: {v}")
            if v == 21:
                if len(self.playerhands[player]) == 2:
                    print("[green1]Natural blackjack![/green1]")
                else:
                    print("[green1]Blackjack![/green1]")
                break
            x = input(f"\n[{player.name}] > ").lower()
            if x in ['1','h','hit']:
                print("Drew:",self.deck.peek()[0])
                self.draw(player)
                if self.value(self.playerhands[player])>21:
                    print("[red]Bust.[/red]")
                    break
            elif x in ['2','s','stand']:
                print("Standing.")
                break
            elif x in ['3','double','2x','double down']:
                print("Doubling.")
                self.bets[player]*=2
                print("Drew:",self.deck.peek()[0])
                self.draw(player)
                print("Hand:")
                print(self.playerhands[player])
                if self.value(self.playerhands[player])>21: print("[red]Bust.[/bust]")
                break
            elif x=="quit":
                quit()
            elif x in ["help",'h','format']:
                print(BLACKJACK_HELP)
                continue
            else:
                print("Invalid input, type 'help' to see the options.")
                continue
        print(SEP)
    def round(self):
        "Loop of a full round."
        print(SEP)
        print(f"\nStarting round #{self.rounds}!")
        # Betting
        for p in self.players:
            col = "green1" if p.money>=0 else "red1"
            print(f"\n{p.name}, you have [{col}]{p.money}$[/{col}].")
            x = input("Place your bet: ")
            try: self.bets[p]=float(x)
            except:
                print("Invalid bet! Defaulting to 100$.")
                self.bets[p] = 100.0
        # Dealing
        print(SEP)
        print("\n[pink1]Dealing cards...[pink1]\n")
        self.deal()
        print("Dealer's hand:",self.dealerhand)
        print(SEP)
        # Player turns
        for p in self.players:
            self.player_turn(p)
        # Dealer's turn
        print("[pink1]The dealer is playing...[/pink1]\n")
        self.dealer_turn()
        print("Dealer's hand:",self.dealerhand)
        # Payout
        print(f"Value: {self.value(self.dealerhand)}")
        print("\nResults:")
        self.payout(info=True)
        self.rounds += 1
    def mainloop(self, players: Player | list | None = None) -> int:
        """Handles a full game. Returns number of played rounds."""
        print("\n[bold]Welcome to Blackjack![/bold]")
        if players is not None: self.init_players(players)
        elif isinstance(players,Player): players = [players]
        players = self.players
        if self.rounds == 1:
            print(SEP)
            print(f"""Player options are as follows. Type 'help' to see them again:{BLACKJACK_HELP}""")
            print(SEP)
        print(f"\nHello {', '.join([p.name for p in players])}! You have played {self.rounds-1} rounds. Want to play{" again" if self.rounds>1 else ''}?")
        while input(f"Play {"again" if self.rounds>1 else '(Enter/X)'}?\n> ").lower() in ['','y','yes','1','sure','ok','play']: 
            self.round()
            print(SEP)
            self.reset()
        if self.rounds == 1:
            print("A shame. Happy gambling!")
        else:
            print("Thanks for playing Blackjack. Come again soon!")
        return self.rounds-1

class UNO:
    def __init__(self, casino: Casino | None = None):
        self.deck = Deck("UNO")
        self.casino = casino
    def __repr__(self):
        return f"UNO({f"casino={self.casino}" if self.casino is not None else ''})"
    def __str__(self):
        return "UNO"

if __name__ == '__main__':
    casino = Casino("Crystal Palace")
    casino.mainloop()
