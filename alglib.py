"""Library for many of the useful data structures and helper functions I made."""

import math
from string import punctuation

# FUNCTIONS

# 1. Functions - Collections

def flatten(nested: list | tuple | set | dict) -> list:
    """Flattens lists, tuples, sets and dicts."""
    flat_list = []
    try:
        for item in nested:
            if isinstance(item, (list, tuple, set, dict)):
                flat_list.extend(flatten(item))
            else:
                flat_list.append(item)
        return flat_list
    except TypeError: return [nested]

def rev_dict(s: dict, duplicates: bool = False) -> dict:
    """Maps values to keys"""
    inverse = dict()
    for k,v in s.items():
        if duplicates: inverse.setdefault(v,[]).append(k)
        else: inverse[v] = k
    return inverse

def remove_duplicates(arr) -> list:
    """Preserves ordering"""
    seen = set()
    return [x for x in arr if not (x in seen or seen.add(x))]

def powerset(s: list):
    """Returns the power set of the list."""
    if not s:
        return [[]]
    elem = s[0]
    rest = s[1:]
    subsets = powerset(rest)
    return subsets + [[elem] + subset for subset in subsets]

def subsets(arr: list, n: int) -> list:
    """Returns all the n-length subsets of the input list."""
    res = []
    def rec(path, i):
        if len(path)==n:
            res.append(path[:])
            return
        if i >= len(arr): return
        rec(path+[arr[i]],i+1)
        rec(path,i+1)
    rec([],0)
    return res

# 2. Functions - Strings

def snake_to_camel(s: str) -> str:
    """snake_case to camelCase"""
    text = s.split("_")
    return text[0] + "".join(x.title() for x in text[1:])

def camel_to_snake(s: str) -> str:
    """camelCase to snake_case"""
    l, r = 0, 0
    text = []
    while r < len(s):
        if s[r].isupper():
            text.append(s[l:r])
            l = r
        r += 1
    text.append(s[l:r+1])
    return "_".join(text).lower()

def is_anagram(a, b):
    t = list(a.lower().replace(" ",""))
    s = list(b.lower().replace(" ",""))
    return sorted(t) == sorted(s)

def is_palindrome(word: str) -> bool:
    word = word.lower().strip("!,.'?;:").replace(" ","")
    return word == word[::-1]

def valid_parenthesis(string: str) -> bool:
    """Checks for invalid parentheses in a string."""
    opening = 0
    for c in string:
        if c == "(":
            opening += 1
        elif c == ")":
            opening -= 1
        if opening < 0: return False
    return opening==0

def checkp(file: str) -> None:
    """Checks for invalid parentheses in file"""
    with open(file,"r", encoding="utf-8") as f:
        path = ""
        curr = 0
        for line in f.readlines():
            curr += 1
            path += line
            if not valid_parenthesis(path):
                print(f"Invalid parenthesis on line {curr}")
                path = ""

# 3. Functions - Numbers / Conversions

def is_prime(n: int) -> bool:
    """Checks whether a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def prime_list(n: int) -> list:
    """Returns a list of all prime numbers from 2 to n"""
    return list(filter(is_prime, [n+1 for n in range(n)]))

def byte_converter(bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024
    return f"{bytes:.2f}YB"

def crgb(color: str) -> tuple:
    """Converts hex colors to RGB."""
    color = color.strip("#").lower()
    if len(color) == 6: return (int(color[:2],16),int(color[2:4],16),int(color[4:6],16))
    elif len(color) == 3: return tuple([int(color[i],16)*17 for i in range(3)])
    else: raise ValueError("invalid hex code")

def chex(color: str | tuple | int, g: int | None = None, b: int = 0) -> str:
    """Converts RGB colors to hex."""
    def to_hex(r,g,b):
        if not (0<=r<=255 and 0<=g<=255 and 0<=b<=255): return None
        return f"#{"".join(hex(x)[2:].rjust(2,"0") for x in [r,g,b])}"
    if isinstance(color,str):
        color = color.strip("rgb() ")
        if "," in color: return to_hex(*map(int,color.split(",")))
        else: return to_hex(*map(int,color.split()))
    elif isinstance(color,tuple): return to_hex(*color)
    elif isinstance(color,int):
        return to_hex(color,color,color) if g is None else to_hex(color,g,b)
    else: raise ValueError("invalid RGB code")

def invert(color: str | tuple, rgb_output: bool = False) -> str:
    """Returns the inverted color. If rgb_output is toggled, always returns an RGB tuple."""
    try:
        if isinstance(color,str) and (color.startswith("#") or "," not in color):
            r,g,b = crgb(color)
            if not rgb_output: return chex(255-r,255-g,255-b)
        else:
            if isinstance(color,tuple): r,g,b = color
            else: r,g,b = map(int,color.strip("rgb() ").split(","))
            if not rgb_output: return f"rgb({255-r},{255-g},{255-b})"
        return (255-r,255-g,255-b)
    except ValueError: raise ValueError("invalid color code")

# 4. Functions - Text analysis

def common_letters(inp: str, info: bool = True, n: int = 0, uppercase: bool = True) -> None | list:
    """Prints the most common letters in order. If info is false, returns an ordered list."""
    letters = {}
    with open(inp, 'r',encoding="utf-8") as f:
        for line in f:
            for c in line.strip().upper() if uppercase else line.strip().lower(): 
                if c.isalpha(): letters[c] = letters.get(c,0) + 1
    if info:
        g = sum(letters.values())/100
        for i,(k,v) in list(enumerate(sorted(letters.items(),key=lambda x: -x[1])))[:n if n>0 else len(letters)]:
            print(f"{i}. {k} | {v} ({v/g:.2f}%)")
    else:
        return list(map(lambda x:x[0],sorted(letters.items(),key=lambda x: x[1],reverse=True)
                    ))[:n if n>0 else len(letters)]

def common_chars(inp: str, info: bool = True, n: int = 0) -> None | list:
    """Prints the most common characters in order. If info is false, returns an ordered list."""
    chars = {}
    with open(inp, 'r',encoding="utf-8") as f:
        for line in f:
            for c in line.strip(): 
                chars[c] = chars.get(c,0) + 1
    if info:
        g = sum(chars.values())/100
        for i,(k,v) in list(enumerate(sorted(chars.items(),key=lambda x: -x[1])))[:n if n>0 else len(chars)]:
            print(f"{i}. {k} | {v} ({v/g:.2f}%)")
    else:
        return list(map(lambda x:x[0],sorted(chars.items(),key=lambda x: -x[1])
                    ))[:n if n>0 else len(chars)]

def common_words(inp: str, info: bool = True, n: int = 0) -> None | list:
    """Prints the most common words in order. If info is false, returns an ordered list."""
    X = str.maketrans(dict(zip(list(punctuation),("" for _ in range(len(punctuation))))))
    words = {}
    with open(inp, 'r',encoding="utf-8") as f:
        for line in f:
            for w in line.strip().split():
                w = w.lower().translate(X)
                words[w] = words.get(w, 0) + 1
    if info:
        g = sum(words.values())/100
        for i,(k,v) in list(enumerate(sorted(words.items(),key=lambda x: (-x[1],x[0]))))[:n if n>0 else len(words)]:
            print(f"{i}. {k} | {v} ({v/g:.2f}%)")
    else:
        return list(map(lambda x:x[0],sorted(words.items(),key=lambda x: (-x[1],x[0]))
                    ))[:n if n>0 else len(words)]


# CLASSES


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