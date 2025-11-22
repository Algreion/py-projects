import random
from collections import deque
from rich import print,text

N2W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten"}
W2N = dict([(w,n) for n,w in N2W.items()])
FACEDOWN = "purple"
RICH = True

class Player:
    def __init__(self, name: str = "Player", money: float = 0, skill: int = 0, age: int = 18):
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

class Casino:
    def __init__(self, name: str = "Casino", money: float = 1e6, dealer: Player | None = None):
        self.name = name
        self.money = money
        self.size = 0
        self.db = dict()
        self.dealer = dealer
    def __repr__(self):
        return f"Casino(name={self.name},money={self.money},players={self.size})"
    def __str__(self):
        return self.name
    def players(self):
        return list(self.db.keys())
    def register(self, players: Player | list):
        if isinstance(players,Player): players = [players]
        for p in players:
            self.db[p] = self.size
            self.size += 1
    def remove(self, players: Player | list):
        if isinstance(players,Player): players = [players]
        for p in players:
            if p in self.db:
                del self.db[p]
                self.size -= 1

class Blackjack:
    def __init__(self, players: Player | list | None, casino: Casino | None = None):
        """States: BETTING | DEAL | PLAYER_TURN | DEALER_TURN | PAYOUT"""
        self.deck = Deck("classic")
        self.casino = casino
        self.dealer = casino.dealer if casino is not None else Player("Dealer")
        self.dealerhand = Hand(self.dealer,"Blackjack")
        self.players = None
        self.rounds = 1
        if players is None:
            self.players = [Player()]
        else:
            if isinstance(players,Player): players = [players]
            self.players = players
        self.playerhands = dict([(player,Hand(player,"Blackjack")) for player in self.players])
        self.bets = dict([(player,0) for player in self.players])
        self.deck.shuffle()
    def __repr__(self):
        return f"Blackjack(players={len(self.players)},rounds={self.rounds},dealer={self.dealer.name})"
    def reset(self) -> None:
        """Resets the round, returning all cards to the deck."""
        self.dealerhand.discard(self.deck)
        for h in self.playerhands.values(): h.discard(self.deck)
        self.bets = {[(player,0) for player in self.players]}
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
            x = self.determine(self.playerhands[p])
            p.money += x*self.bets[p]
            if info and x!=0: print(f"{p.name} {"wins" if int(x>0) else "loses"} {abs(x*self.bets[p])}$!")
            if self.casino is not None: self.casino.money -= x*self.bets[p]
            p.wins += int(x>0)
            p.losses += int(x<0)
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
            if info:  print("Tie.")
            return 0
    def player_turn(self, player: Player):
        """Handles an individual player's turn."""
        print(f"{player.name}'s turn.\n\nDealer's hand:")
        print(self.dealerhand)
        while True:
            print(f"\n{player.name}'s hand:")
            print(self.playerhands[player])
            v = self.value(self.playerhands[player])
            print(f"Value: {v}")
            if v == 21:
                print("[green1]Natural blackjack![/green1]")
                break
            x = input("""\nOptions:
0 / H - Hit
1 / S - Stand
2 / D - Double down (x2 bet, take 1 card, stand)""").lower()
            if x in ['0','h','hit']:
                print("Drew:",self.deck.peek()[0])
                self.draw(player)
                if self.value(self.playerhands[player])>21:
                    print("[red]Bust.[/red]")
                    break
            elif x in ['1','s','stand']:
                print("Standing.")
                break
            elif x in ['2','double','2x','double down']:
                print("Doubling.")
                self.bets[player]*=2
                print("Drew:",self.deck.peek()[0])
                self.draw(player)
                print("Hand:")
                print(self.playerhands[player])
                if self.value(self.playerhands[player])>21: print("[red]Bust.[/bust]")
                break
            else:
                print("Invalid input, try again.")
                continue
    def round(self):
        "Loop of a full round."
        print(f"Starting round #{self.rounds}!")
        # Betting
        for p in self.players:
            x = input(f"\n{p.name}, you have {p.money}$.\nPlace your bet: ")
            self.bets[p]=float(x)
        # Dealing
        print("\nDealing cards...\n")
        self.deal()
        print("Dealer's hand:",self.dealerhand)
        # Player turns
        for p in self.players():
            self.player_turn(p)
        # Dealer's turn
        print("\nThe dealer is playing...")
        print("Dealer's hand:",self.dealerhand)
        # Payout
        print(f"Dealer's value: {self.value(self.dealerhand)}")
        print("Results:")
        self.payout(info=True)
        self.rounds += 1
    def mainloop(self, players: Player | list | None = None):
        print("Welcome to Blackjack!")
        if players is None:
            players = Player(money=100)
        if isinstance(players,Player): players = [players]
        print(f"Hello {', '.join([p.name for p in players])}! You have played {self.rounds-1} rounds. Want to play{" again" if self.rounds>1 else ''}?")
        while input("(Y/N or 1/0) > ").lower() in ['y','1',"yes"]: self.round()
        if self.rounds == 1:
            print("A shame. Happy gambling!")
        else:
            print("Thanks for playing. Come again soon!")

class UNO:
    def __init__(self, ):
        self.deck = Deck("UNO")
    def __repr__(self):
        return f"UNO()"
