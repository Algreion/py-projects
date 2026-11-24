from collections import defaultdict
import random

class MinHeap:
    def __init__(self, heap: list = []):
        """Data structure that always keep minimum value on top."""
        self.heap = []
        self.insertmulti(heap)

    def __str__(self):
        return str(self.heap)
    def __repr__(self):
        return f"MinHeap(min={self.peek() if len(self.heap)!=0 else None},size={self.size()})"
    def __getitem__(self, index: int):
        return self.heap[index]
    def __setitem__(self, index: int, value):
        self.heap[index] = value
    def __iter__(self):
        return iter(self.heap)
    def __len__(self):
        return len(self.heap)
        
    def insert(self, val):
        self.heap.append(val)
        self.heapify_up(len(self.heap)-1)

    def insertmulti(self, values: list):
        for value in values:
            self.insert(value)

    def heapify_up(self, index):
        while index > 0:
            parent = (index-1) // 2
            if self.heap[index] < self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else:
                break

    def heappop(self):
        """Returns minimum value (root)"""
        if len(self.heap) == 0: raise IndexError("Empty heap")
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.heapify_down(0)
        return min_val
    
    def pushpop(self, val):
        """Push item on the heap, then pop and return the smallest item"""
        self.insert(val)
        return self.heappop()

    def heapify_down(self, index):
        while True:
            left_child = 2*index+1
            right_child = 2*index+2
            smallest = index
            if left_child < len(self.heap) and self.heap[left_child] < smallest:
                smallest = left_child
            if right_child < len(self.heap) and self.heap[right_child] < self.heap[smallest]:
                smallest = right_child
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break
            
    def peek(self):
        if len(self.heap) == 0: raise IndexError("Peek from an empty heap")
        return self.heap[0]
    
class MaxHeap:
    def __init__(self, heap: list = []):
        """Data structure that always keep maximum value on top."""
        self.heap = []
        self.insertmulti(heap)

    def __str__(self):
        return str(self.heap)
    def __repr__(self):
        return f"MaxHeap(max={self.peek() if len(self.heap)!=0 else None},size={self.size()})"
    def __getitem__(self, index: int):
        return self.heap[index]
    def __setitem__(self, index: int, value):
        self.heap[index] = value
    def __iter__(self):
        return iter(self.heap)
    def __len__(self):
        return len(self.heap)
    
    def insert(self, val):
        self.heap.append(val)
        self.heapify_up(len(self.heap)-1)
    
    def insertmulti(self, values: list):
        for v in values:
            self.insert(v)
    
    def heappop(self):
        if len(self.heap) == 0: raise IndexError("Pop from an empty heap")
        max_val = self.heap[0]
        self.heap[0], self.heap[-1] == self.heap[-1], self.heap[0]
        self.heap.pop()
        self.heapify_down(0)
        return max_val

    def pushpop(self, val):
        """Push value on the heap, then pop and return the largest value"""
        self.insert(val)
        return self.heappop()
    
    def heapify_up(self, index):
        while index > 0:
            parent = (index-1) // 2
            if self.heap[index] > self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else: 
                break
    
    def heapify_down(self, index):
        while True:
            left_child = (index*2) + 1
            right_child = (index*2) + 2
            largest = index
            if right_child < len(self.heap) and self.heap[largest] < self.heap[right_child]:
                largest = right_child
            if left_child < len(self.heap) and self.heap[largest] < self.heap[left_child]:
                largest = left_child
            if largest != index:
                self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
                index = largest
            else: 
                break
    
    def peek(self):
        if len(self.heap) == 0: raise IndexError("Peek from an empty heap")
        return self.heap[0]

def flatten(nested):
    flat_list = []
    try:
        for item in nested:
            if isinstance(item, (list, tuple, set, dict)):
                flat_list.extend(flatten(item))
            else:
                flat_list.append(item)
        return flat_list
    except:
        return None

class Hashset:
    def __init__(self, prev: list = [], capacity: int = 100):
        """Hashset datastructure, mimicking a python set."""
        self._capacity = capacity
        self._bins = [[] for _ in range(self._capacity)]
        self._size = 0
        self.addmulti(*prev)
    
    def __len__(self):
        return self._size
    def __add__(self, items) -> int:
        return self.addmulti(items)
    def __sub__(self, items):
        return self.removemulti(items)
    def __str__(self):
        return f"set{str(tuple(self.read()))}"
    def __repr__(self):
        return "{"+f"{", ".join(str(v) for v in self.read())}"+"}"
    def __getitem__(self, index: int):
        return self.read()[index]
    def __delitem__(self, value):
        self.remove(value)
    def __iter__(self):
        return iter(self.read())

    def hashf(self, item) -> int:
        return hash(item) % self._capacity
    
    def add(self, item) -> bool:
        if item == None:
            return False
        if self._size >= self._capacity:
            self.resize()
        try:
            index = self.hashf(item)
        except TypeError:
            return False
        bucket = self._bins[index]
        for v in bucket:
            if v == item:
                return False
        bucket.append(item)
        self._size += 1
        return True
    
    def addmulti(self, *items) -> int:
        res = 0
        for item in items:
            if isinstance(item, (tuple,list,set,dict)):
                item = flatten(item)
                for i in item:
                    if self.add(i): res += 1
            elif isinstance(item, Hashset):
                for i in item.read():
                    if self.add(i): res += 1
            else:
                if self.add(item): res += 1
        return res
    
    def check(self, item):
        index = self.hashf(item)
        bucket = self._bins[index]
        for i in bucket:
            if i == item:
                return True
        return False
    
    def remove(self, item) -> int:
        index = self.hashf(item)
        bucket = self._bins[index]
        for i in range(len(bucket)):
            if bucket[i] == item:
                del self._bins[index][i]
                self._size -= 1
                return 1
        return 0
    
    def removemulti(self, *items) -> int:
        deleted = 0
        if isinstance(items[0], Hashset):
            for i in items[0].read():
                if self.remove(i): deleted += 1
        items = flatten(items)
        for item in items:
            if self.remove(item): deleted += 1
        return deleted
    
    def resize(self) -> bool:
        old_buckets = self._bins
        self._capacity = int(self._capacity*1.5)
        self._bins = [[] for _ in range(self._capacity)]
        for bucket in old_buckets:
            for item in bucket:
                self._bins[hash(item) % self._capacity].append(item)

    def clear(self, confirm: int = 100)-> int:
        if confirm and self._size >= confirm:
            confirm = input(f"""Are you sure? You are about to delete {self._size} items. Type "y" to confirm, else cancel: """)
            if confirm != "y":
                return 0
        temp = self._size
        self._bins = [[] for _ in range(self._capacity)]
        self._size = 0
        return temp
    
    def read(self) -> list:
        res = []
        for bucket in self._bins:
            for item in bucket:
                res.append(item)
        return res
    
    def intersection(self, other: object) -> object:
        result = Hashset()
        for item in self.read():
            if other.check(item):
                result.add(item)
        return result
    
    def union(self, other: object) -> object:
        res = Hashset()
        res + other
        res + self
        return res

class Hashmap:
    def __init__(self, prev: list = [], size=16):
        """Hashmap data structure, mimicking a python dictionary."""
        self.size = size
        self.buckets = [[] for _ in range(self.size)]
        self.num = 0
        if type(prev)==dict: prev = prev.items()
        self.update(prev)

    def __repr__(self):
        return "{"+", ".join(", ".join(f"{str(k)}: {str(v)}" for k,v in b) for b in self.buckets if b)+"}"
    def __str__(self):
        return self.__repr__()
    def __getitem__(self, key):
        return self.get(key)
    def __setitem__(self, key, value):
        self.add(key, value)
    def __delitem__(self, key):
        self.delete(key)
    def __len__(self):
        return self.num
    def __iter__(self):
        return iter(self.keys())

    def hashf(self, key) -> int:
        return hash(key) % self.size

    def add(self, key, value):
        if self.num == self.size:
            self.resize()
        index = self.hashf(key)
        bucket = self.buckets[index]
        for i,(k,_) in enumerate(bucket):
            if k == key:
                bucket[i] = (key,value)
                return
        bucket.append((key,value))
        self.num += 1

    def get(self, key):
        index = self.hashf(key)
        bucket = self.buckets[index]
        for k,v in bucket:
            if k == key:
                return v
        return None

    def delete(self, key) -> bool:
        index = self.hashf(key)
        bucket = self.buckets[index]
        for i,(k,_) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return True
        return False

    def resize(self) -> None:
        self.size *= 2
        old_buckets = self.buckets
        self.buckets = [[] for _ in range (self.size)]
        for bucket in old_buckets:
            for k,v in bucket:
                index = self.hashf(k)
                self.buckets[index].append((k,v))
    
    def keys(self) -> list:
        """Returns a list of keys in the hashmap."""
        res = []
        for b in self.buckets:
            for k,_ in b:
                res.append(k)
        return res
    def values(self) -> list:
        """Returns a list of values in the hashmap."""
        res = []
        for b in self.buckets:
            for _,v in b:
                res.append(v)
        return res
    def items(self) -> list:
        """Returns a list of (key,value) pairs"""
        res = []
        for b in self.buckets:
            for k,v in b:
                res.append((k,v))
        return res
    
    def update(self, new: list):
        """Updates the hashmap with key,value pairs as tuples."""
        for k,v in new:
            self[k] = v

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
    def change(self, v1, v2):
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
    
class Node:
    def __init__(self, data = None, prev = None, nxt = None):
        self.data = data
        self.next = nxt
        self.prev = prev
    def __repr__(self):
        return f'Node(data={self.data},prev={self.prev.data if self.prev else None},next={self.next.data if self.next else None})'

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

def guessing(n: int = None): # Guessing game
    if n is not None:
        try: 
            n = int(input("Welcome to the number guessing game. Choose the maximum number that it could be: "))
            if 0 <= n < 2:
                print("You won in 0 guesses, yay..")
                return
        except (Exception, KeyboardInterrupt):
            print("Game cancelled")
            return
    rand = random.randint(1,n) if n > 1 else random.randint(n,0)
    guesses = 1
    while True:
        guess = input("\nWhat is your guess? ")
        if not guess.isdigit():
            print("Your guess should be a number!")
            continue
        guess = int(guess)
        if guess > rand:
            print(f"Your guess of {guess} was greater than the target.")
        elif guess < rand:
            print(f"Your guess of {guess} was lower than the target!")
        else:
            print(f"You won, the answer was indeed {guess}! You took {guesses} attempts to guess correctly! (amount = {abs(n)})")
            if guesses == 1 and n >= 10:
                print("Impressive, great luck!")
            if guesses <= 3 and n >= 100:
                print("Wow, great gaming chair!")
            if guesses > 100:
                print("Why did you spend so much time on this...")
            break
        guesses += 1
def guesser(n): # Guesses a random number
    if n <= 1:
        print(f"The answer was {n}. I took 1 attempts. Number pool: 0.")
        return
    rand = random.randint(1,n) if n > 1 else random.randint(n,0)
    guesses = 1
    guess = random.randint(1,n) if n > 1 else random.randint(n,0)
    if n > 0:
        l, r = 1, n
    else:
        l, r = n, 0
    while True:
        print(guess,end=" ") if guesses % 2 else print(guess)
        if guess == rand:
            print(f"\nThe answer was {guess}. I took {guesses} attempts. Number pool: {n}.")
            return
        if guess > rand:
            r = guess-1
        else:
            l = guess+1
        guess = (l+r)//2
        guesses += 1
def guesser2(n, range: int = 6): # Guesses the chosen number
    range = 10**range
    req = n
    guesses = 1
    if n > 0:
        while n > range: range *= 10
        l, r = 1, range
    else:
        range *= -1
        while n < range: range *= 10
        l, r = range, 0
    guess = random.randint(1,range) if range > 0 else random.randint(range,0)
    while True:
        print(guess,end=" ") if guesses % 2 else print(guess)
        if guess == req:
            print(f"\nThe answer was {guess}. I took {guesses} attempts. Number pool: {range}.")
            return
        if guess > req:
            r = guess-1
        else:
            l = guess+1
        guess = (l+r)//2
        guesses += 1

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

# SHORTCUTS
# Ctrl+Alt+(^/v): Add cursor above/below
# Shift+Alt+(^/v): Copy this line above/below
# Alt+(^/v): Shift this line above/below
# Ctrl+Shift+K: Delete this line
# Ctrl+K + Ctrl+C: Comment out
# Ctrl+K + Ctrl+U: Uncomment
# Ctrl+D: Select next match (if already selecting something)
# Ctrl+U: Undo cursor placement
# Ctrl+L: Select whole line
# Ctrl+ ì: Indent line
# Ctrl+ ': Outdent line
# Ctrl+Shift+Enter: Insert line above
# Ctrl+Enter: Insert line above
# Ctrl+D: Select the word, again to select next instance
# Ctrl+J: Toggle terminal
# Ctrl+ò: Focus on terminal