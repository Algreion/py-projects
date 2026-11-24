"""Library for the experimental classes and data structures I made."""

from collections import defaultdict

class User:
    def __init__(self, name: str, age: int, username: str = None):
        self.name = name
        self.age = age
        self.username = "@"+username if username else "@"+name.lower()
        self.banned = False
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"{self.name.capitalize()}(age={self.age}, user={self.username}, ban={str(int(self.banned))})"
    def greet(self):
        print(f"> {self.username}: Hello, I am {self.name}!" if not self.banned else "")
    def chat(self, msg: str = ''):
        print(f"> {self.username}: {msg}" if not self.banned else "")
    def check18(self):
        return self.age >= 18 and not self.banned
    def data(self):
        return {"name": self.name, "username": self.username, "age": self.age, "banned": self.banned}

def populate() -> object:
    server = LinkedList([User("YHVH", float("inf"), "allfather")])
    users = [
                User("Alice", 17, "alice123"),
                User("Bob", 25, "bob_the_builder"),
                User("Charlie", 35, "charlie_brown"),
                User("Diana", 28, "wonder_woman"),
                User("Eve", 22, "eve_online"),
                User("Frodo", 33, "ring_bearer"),
                User("Gandalf", 2019, "the_grey"),
                User("Harry", 11, "the_boy_who_lived"),
                User("Igor", 40, "mad_scientist"),
                User("John", 45, "john_doe")
            ]
    server.addmulti(users)
    return server

class Twitter:
    def __init__(self):
        self.time = 0
        self.users = set()
        self.followMap = defaultdict(set)
        self.tweetMap = defaultdict(list)
    def __repr__(self):
        return f'Twitter(time={self.time},users={len(self.users)})'

    def postTweet(self, userId: int, tweet: str) -> None:
        self.users.add(userId)
        self.tweetMap[userId].append((self.time, tweet))
        self.time += 1

    def getNewsFeed(self, userId: int, n: int = 10) -> list:
        self.users.add(userId)
        feed = self.tweetMap[userId][:]
        for followee in self.followMap[userId]:
            feed.extend(self.tweetMap[followee][:])
        return [tweet for _, tweet in sorted(feed, reverse=True)[:n]]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.users.add(followerId)
        self.users.add(followeeId)
        if followerId != followeeId:
            self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.users.add(followerId)
        self.users.add(followeeId)
        self.followMap[followerId].discard(followeeId)

class Keyboard:
    def __init__(self):
        kb = """!"£$%&/()=?\nqwertyuiop\nasdfghjkl\nzxcvbnm,.\n"""
        self.kb = [list(line) for line in kb.splitlines()]
        self.ab = set(kb.replace('\n','')+' ')
        self.dist = self.maps()
    
    def maps(self) -> dict:
        dists = dict()
        for i in range(len(self.kb)):
            for j in range(len(self.kb[i])):
                spc = round(((i-len(self.kb))**2+min([(j+0.5*i-x)**2 for x in [i+.5 for i in range(4,7)]]))**0.5)
                dists[(self.kb[i][j],' ')] = spc
                dists[(' ',self.kb[i][j])] = spc
                for i0 in range(len(self.kb)):
                    for j0 in range(len(self.kb[i0])):
                        if (x:=(self.kb[i0][j0],self.kb[i][j])) in dists:
                            dists[(self.kb[i][j],self.kb[i0][j0])] = dists[x]
                        dists[(self.kb[i][j],self.kb[i0][j0])] = round(((i-i0)**2+((j+0.5*i)-(j0+0.5*i0))**2)**0.5,2)
        return dists
    
    def distance(self, char1: str, char2: str) -> float:
        try: return self.dist[(char1,char2)]
        except: return 0
    
    def closest(self, char: str, n: int = 3, onlyLetters: bool = True) -> list:
        if char not in self.ab: raise ValueError(f'Character "{char}" not in keyboard.')
        return sorted([(c,self.dist[(char,c)]) for c in filter(lambda x: x.isalpha() if onlyLetters else True, self.ab) if c != char], key=lambda x: x[1])[:n]
    def farthest(self, char: str, n: int = 3, onlyLetters: bool = True) -> list:
        if char not in self.ab: raise ValueError(f'Character "{char}" not in keyboard.')
        return sorted([(c,self.dist[(char,c)]) for c in filter(lambda x: x.isalpha() if onlyLetters else True, self.ab) if c != char], key=lambda x: -x[1])[:n]
    
    def eval(self, word: str) -> float:
        d = 0
        word = word.lower()
        for a,b in zip(word,word[1:]):
            d += self.distance(a,b)
        return d

    def evalFile(self, file: str = 'words.txt') -> dict:
        words = dict()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                for w in line.split():
                    if w not in words: words[w] = self.eval(w)
        return words
    def evalText(self, txt: str) -> dict:
        words = dict()
        for line in txt.splitlines():
            for w in line.split():
                if w not in words: words[w] = self.eval(w)
        return words
    def rankFile(self, file: str = 'words.txt', n: int = 10, desc: bool = True) -> list:
        words = self.evalFile(file)
        desc = -1 if desc else 1
        return [w for w,_ in sorted(words.items(), key=lambda x: desc*x[1])[:n]]
    def rankText(self, txt: str = 'words.txt', n: int = 10, desc: bool = True) -> list:
        words = self.evalText(txt)
        desc = -1 if desc else 1
        return [w for w,_ in sorted(words.items(), key=lambda x: desc*x[1])[:n]]


def solveNQueens(n):
    res = []
    left_diagonal = set()
    right_diagonal = set()
    cols = set()
    def valid(row, col):
        return not (col in cols or (row-col) in left_diagonal or (row+col) in right_diagonal)
    def backtrack(board, row):
        if row == n:
            copy = []
            for i in range(len(board)):
                copy.append("".join(board[i]))
            res.append(copy)
            return
        for col in range(n):
            if valid(row, col):
                board[row][col] = "Q"
                cols.add(col)
                left_diagonal.add(row-col)
                right_diagonal.add(row+col)
                backtrack(board,row+1)
                board[row][col] = "."
                cols.remove(col)
                left_diagonal.remove(row-col)
                right_diagonal.remove(row+col)
    board = [["."]*n for _ in range(n)]
    backtrack(board,0)
    return res

def todo_list(todo=[]):
    def menu():
        print("\nTo-Do List")
        print("1. Add a task")
        print("2. View tasks")
        print("3. Remove a task")
        print("4. Exit")
    while True:
        menu()
        choice = input("Choose an option (1-4): ")
        if choice == "1":
            task = input("Enter the task: ")
            todo.append(task)
            print(f"""Task "{task}" added.""")
        elif choice == "2":
            if not todo:
                print("\nNo tasks left.")
                continue
            print("\n")
            for i, t in enumerate(todo):
                print(f"""{i+1}. "{t}" """)
        elif choice == "3":
            if not todo: 
                print("No tasks to delete.")
                continue
            task = int(input("Enter the number of the task to remove: "))-1
            if task >= len(todo): print("Invalid, that task doesn't exist.")
            else:
                task = todo.pop(task)
                print(f"""Task "{task}" removed.""")
        elif choice == "4":
            print("Goodbye!")
            break
        elif choice == '': break
        else: print("Invalid choice. Please try again.")
