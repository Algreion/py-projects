"""Library for the experimental classes and data structures I made."""

from collections import defaultdict
from contextlib import contextmanager
import random
import time
import re


# FUNCTIONS

# 1. Functions - Encoding

def to_morse(text: str) -> str:
    morse_code = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '"': '.-..-.',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----', ',': '--..--', '.': '.-.-.-', '=': '-...-', '+': '.-.-.', '-': '-....-',
    '?': '..--..', '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-', '!': '-.-.--', "'": '.----.', '&': '.-...', '$': '...-..-', ' ': '/'
    }
    return " ".join(morse_code.get(c, "") for c in text.upper())

def from_morse(morse: str) -> str:
    morse_code_rev = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-..-.': '"',
    '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '.----.': "'", '.-...': '&', '...-..-': '$',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7', '-.-.--': '!', '-...-': '=', '.-.-.': '+' , '-....-': '-',
    '---..': '8', '----.': '9', '-----': '0', '--..--': ',', '.-.-.-': '.', '..--..': '?', '-..-.': '/', '-....-': '-', '-.--.': '(', '-.--.-': ')', '/': ' '
    }
    return "".join(morse_code_rev.get(c, "") for c in morse.split())

def caesar_cipher(text: str, n: int = 10, back: bool = False) -> str:
    """Returns a simple crypted string. Back is for deciphering"""
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    res = ""
    n = -n if back else n
    for i in range(len(text)):
        res += alpha[(alpha.index(c) + n) % len(alpha)] if (c:=text[i]) in alpha else c
    return res

# 2. Functions - Puzzles & Algorithms

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

def solveGoogle(arr, k):
    """Google Kick Start competition - Round C 2020
    Problem: Given an array of N integers, a contiguous subarray is an m-countdown if
    it is of length m and contains the integers m, m-1, ..., 2, 1 in that order.
    For example, [3,2,1] is a 3-countdown. Return the number of K-countdowns in the array."""
    n = 0
    curr = k
    for i in range(len(arr)):
        if arr[i] == k:
            curr = k
        if curr > -1:
            if arr[i] == curr: curr-=1
            else: curr = -1
        if curr == 0:
            n += 1
            curr = -1
    return n

def solveChange(n: float, info: bool = False) -> int:
    """Returns the lowest number of € coins you need to reach the amount."""
    n = int(n*100)
    coins = [200,100,50,20,10,5,2,1]
    c = []
    res = 0
    while n > 0:
        for i in range(len(coins)):
            if n >= coins[i]:
                n -= coins[i]
                if info: c.append(coins[i]/100)
                res += 1
                break
    return c if info else res

# 3. Functions - Text analysis

def grade(txt: str) -> int:
    """Returns the grade level of readibility of a text, from 0 to 16."""
    # 0.0588 * L - 0.296 * S - 15.8
    s = l = 0
    words = 0
    w = r"[^a-zA-Z]"
    p = r"[!.?]"
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        l += len(re.sub(w,"",line))
        s += len(re.findall(p,line))
        words += len(line.split())
    l = (l/words)*100
    s = (s/words)*100
    grade = round(0.0588*l - 0.296*s - 15.8)
    return 0 if grade < 1 else min(grade, 16)

def grade_file(file: str) -> int:
    """Returns the grade level of a text file."""
    with open(file,'r', encoding='utf-8') as f:
        return grade(f.read())

# 4. Functions - Math & Simulations

def fibonacci(n: int) -> int:
    a,b = 1,1
    f = 0
    if n <= 2:
        return a
    for _ in range(n-2):
        f = a+b
        a = b
        b = f
    return f

def fibonacci_sequence(n: int) -> list:
    s = []
    for i in range(n):
        s.append(fibonacci(i+1))
    return s

def derivative(func, x, h: float = 0.0001, precision: int = 3):
    return round((func(x+h)-func(x))/h, precision)

def birthday_paradox(n: int, attempts: int = 1000, original: bool = False) -> float:
    """Tests the probablity of two people sharing a birthday with n people"""
    matches = 0
    if original:
        for _ in range(attempts):
            s = [random.randint(1,365) for _ in range(n)]
            s.sort()
            for _ in range(n):
                x = s.pop()
                if x in s:
                    matches += 1
                    break
    else:
        for _ in range(attempts):
            s = [random.randint(1,365) for _ in range(n)]
            matches += len(s)!=len(set(s))
    return (matches / attempts * 100)

def birthday_paradox_probability(prob: float) -> int:
    """Returns the number of people to get the % chance of sharing a birthday."""
    l, r = 2, 100
    while l <= r:
        mid = (r+l) // 2
        check = birthday_paradox(mid)
        if int(check) == int(check): return mid
        if check > prob:
            r = mid - 1
        else:
            l = mid + 1
    return mid

def calc_weekday(day: int, month: int, year: int) -> int:
    """Returns the weekday as an integer: 0 Sunday - 6 Saturday"""
    jan = 3 if year%4 else 4
    feb = 28 if year%4 else 29
    doomsdays = {1: jan, 2: feb, 3: 14, 4: 4, 5: 9, 6: 6, 7: 11, 8: 8, 9: 5, 10: 10, 11: 7, 12: 12}
    century = year//100
    if not century % 4: century = 2
    elif not (century-1) % 4: century = 0
    elif not (century+1) % 4: century = 3
    else: century = 5
    d = year % 100
    d = century + d + d//4
    date = (day - doomsdays[month] + d) % 7
    return date

def weekday(day: int, month: int, year: int = 2025) -> str:
    days = {0: 'Saturday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    return days[calc_weekday(day,month,year)]

# 5. Functions - Utility

@contextmanager # "with timer(): \n ..."
def timer():
    start = time.perf_counter()
    start_ns = time.perf_counter_ns()
    yield
    if (x:=time.perf_counter()-start)>0.01: print(f"The operation took {x:.3f} s.")
    elif x*1000>0.1: print(f"The operation took {x*1000:.2f} ms.")
    else: print(f"The operation took {time.perf_counter_ns()-start_ns} ns.")

def filetype(filename: str) -> str:
    """Returns the type of the file given its path."""
    if '.' not in filename or filename.startswith('.'):
        return ".txt"
    return '.' + filename.split('.')[-1]

def filepath(filename: str) -> str:
    """Returns the name of the file given its path."""
    path = ""
    for c in filename.removesuffix(filetype(filename))[::-1]:
        if c == "\\" or c == "/": break
        path += c
    return path[::-1]

def backup(filename: str = None, backup: str = None) -> bool:
    """
    Backup the current file or another file.
    filename: Name of file to backup. Defaults to current file.
    backup: Name of new backup file. Optional.
    """
    if not backup and not filename: print(backup := filepath(__file__)+"_backup.py")
    elif not backup: print(backup := filepath(filename)+"_backup"+filetype(filename))
    if "." not in backup:
        backup += filetype(filename)
        print(backup)
    try:
        if not filename: filename = __file__
        with open(filename, "r", encoding="utf-8") as f:
            with open(backup, "a", encoding="utf-8") as b:
                for line in f:
                    b.write(line)
        return True
    except PermissionError:
        print(f"Permission denied accessing {backup}")
        return False
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        return False

def reorder_file(inp: str, copy: bool = False):
    if copy: backup(inp)
    with open(inp,'r',encoding='utf-8') as f:
        lines = f.readlines()
    with open(inp,'w',encoding='utf-8') as f:
        for line in sorted(lines):
            f.write(line)


# CLASSES


class Node:
    def __init__(self, data = None, prev = None, nxt = None):
        self.data = data
        self.next = nxt
        self.prev = prev
    def __repr__(self):
        return f'Node(data={self.data},prev={self.prev.data if self.prev else None},next={self.next.data if self.next else None})'

class LinkedList:
    def __init__(self, array: list = None):
        """Doubly linked list data structure."""
        self.head = Node()
        self.tail = self.head
        self._size = 0
        if array:
            self.addmulti(array)
    def __str__(self) -> str:
        curr, res = self.head, []
        while curr:
            if curr.data is not None: res.append(str(curr.data)) 
            curr = curr.next
        return ", ".join(res)
    def __repr__(self):
        return f"<<{str(self)}>>"
    def __getitem__(self, index: int):
        return self.get(index)
    def __setitem__(self, index: int, data):
        self.insert(index, data)
    def __delitem__(self, index):
        self.delete_i(index)
    def __add__(self, n):
        if type(n) == LinkedList:
            self.tail.next = n.head
            n.head.prev = self.tail
        else:
            self.append(n)
        return self
    def __iter__(self):
        return iter(self.tolist())
    def __len__(self):
        return self._size

    def _callnodes(self, func):
        """Call a function on each of the nodes"""
        curr = self.head
        while curr:
            func(curr)
            curr = curr.next
    def append(self, data):
        """Add a node at the end of the list"""
        self.tail.next = Node(data, self.tail, None)
        self.tail = self.tail.next
        self._size += 1
    def prepend(self, data):
        """Add a node at the start of the list"""
        self.head.prev = Node(None, None, self.head)
        self.head.data = data
        self.head = self.head.prev
        self._size += 1
    def insert(self, index, data):
        """Inserts a node at a specific position"""
        if index > self._size: index = -1
        if index == -1 or index == self.getsize():
            self.append(data)
        elif index == 0:
            self.prepend(data)
        else:
            node = self.get(index)
            node.prev = Node(data, node.prev, node)
            node.prev.prev.next = node.prev
            self._size += 1
    def search(self, data) -> bool:
        """Returns true if a node with the data exists"""
        curr = self.head
        while curr:
            if curr.data == data: return True
            curr = curr.next
        return False
    def find(self, data):
        """Returns the first node containing the data"""
        curr = self.head
        while curr:
            if curr.data == data: return curr
            curr = curr.next
        return None
    def get(self, index):
        """Returns the node at index i"""
        if index < 0: index = len(self)+index
        elif index >= len(self): return self.tail
        curr = self.head
        while curr and index>=0:
            curr = curr.next
            index -= 1
        return curr
    def addmulti(self, data: list, prepend: bool = False):
        """Adds all the data to the list. Appends by default, prepends otherwise"""
        if prepend:
            for v in data: self.prepend(v)
        else: 
            for v in data: self.append(v)
    def delete(self, data = None, n = None) -> bool:
        """Deletes the data from the list, or the specific node n"""
        if n: node = n
        elif not (node := self.find(data)) or self.head == self.tail: return False
        if node.next:
            node.next.prev = node.prev
        else: self.tail = node.prev
        if node.prev:
            node.prev.next = node.next
        else: self.head = node.next
        del node
        self._size -= 1
        return True
    def delete_i(self, index) -> bool:
        """Deletes the node at position i"""
        return self.delete(None,self.get(index))
    def deletemulti(self, data: list) -> int:
        """Deletes all nodes with the data in list"""
        r = 0
        for v in data: 
            r += self.delete(v)
        return r
    def deletemulti_i(self, start: int, end: int) -> int:
        """Delete all nodes between a specified range"""
        r = 0
        for i in range(start, end):
            r += self.delete_i(i-r)
        return r
    def solve_(self, v1, v2):
        """Change a node's data to a new value"""
        r = 0
        while (node := self.find(v1)):
            node.data = v2
            r += 1
        return r
    def getsize(self): return self._size
    def reverse(self):
        """Reverses the list"""
        curr, self.head, self.tail = self.head, self.tail, self.head
        while curr:
            curr.prev, curr.next, curr = curr.next, curr.prev, curr.next
    def tolist(self) -> list:
        """Return a list representation of the data structure"""
        res = []
        curr = self.head
        while curr:
            if curr.data is not None: res.append(curr.data)
            curr = curr.next
        return res
    def call(self, func):
        """Call a function on every nodes' data"""
        curr = self.head
        while curr:
            func(curr.data)
            curr = curr.next

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