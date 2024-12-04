# https://adventofcode.com/2024/day/1
import time

def d1_distance_solve(arr):
    arr1, arr2 = arr
    diff = 0
    res = zip(sorted(arr1),sorted(arr2))
    for i in res:
        diff += abs(i[0]-i[1])
    return diff

def d1_parse(string):
    arr1 = []
    arr2 = []
    for i,n in enumerate(string.split()):
        if not i%2: arr1.append(int(n))
        else: arr2.append(int(n))
    return [arr1,arr2]

def day1_solve1(input: str) -> int: # Ans: 3714264
    return d1_distance_solve(d1_parse(input))

def day1_solve2(input: str) -> int: # Ans: 18805872
    start = time.perf_counter()
    n = 0
    arr1, arr2 = d1_parse(input)
    hist = {}
    for v in arr2:
        hist[v] = hist.get(v, 0) + 1
    for v in arr1:
        n += v*hist.get(v,0)
    print (time.perf_counter()-start)
    return n

def d2_parse(string):
    x = string.split("\n")
    for i in range(len(x)):
        x[i] = x[i].split()
        for j,n in enumerate(x[i]):
            x[i][j] = int(n)
    return x

def day2_solve1(input: str, split=True) -> int: # Ans: 279
    if split: input = d2_parse(input)
    n = 0
    for report in input:
        safe, i, increase = 1, 0, 0
        while i < len(report)-1 and safe:
            v = report[i]
            if i == 0:
                if report[i+1] == v:
                    safe = 0
                    break
                elif report[i+1] > v:
                    increase = 1
            if increase:
                if report[i+1] <= v or report[i+1]-3 > v: safe = 0
            else:
                if report[i+1] >= v or report[i+1]+3 < v: safe = 0
            i += 1
        n += safe
    return n

def day2_solve2(input: str) -> int: # Ans: 343
    n = 0
    input = d2_parse(input)
    for report in input:
        if day2_solve1([report], False):
            n += 1
            continue
        for i in range(len(report)):
            if day2_solve1([report[:i]+report[i+1:]], False):
                n += 1
                break
    return n

def day3_solve1(input: str, do=False) -> int: # Ans: 181345830
    ruleset = {"m":"u","u":"l","l":"("}
    i, n = 0, 0
    path = ""
    on = True
    while i < len(input):
        c = input[i]
        if on and path:
            if (path[-1] in ruleset and ruleset[path[-1]] == c) or (path[-1] == "(" and c.isdigit()) or (path[-1] == "," and c.isdigit()):
                path += c
            elif path[-1].isdigit():
                if "," in path and (c == ")" or c.isdigit()):
                    path += c
                elif "," not in path and (c.isdigit() or c == ","):
                    path += c
                else: path = ""
            else:
                path = ""
        elif on and c == "m": path += c
        if on and path and path[-1] == ")":
            path = path.strip("mul()").split(",")
            n += int(path[0])*int(path[1])
            path = ""
        if do and c == "d":
            on = True if input[i:i+4] == "do()" else False if input[i:i+7] == "don't()" else on
        i += 1
    return n

def day3_solve2(input: str) -> int: # Ans: 98729041
    return day3_solve1(input, True)

def d4_parse(string):
    res, add = [], []
    for c in string:
        if c == "\n":
            res.append(add[:])
            add.clear()
        else: add.append(c)
    res.append(add)
    return res

def day4_solve1(input: str) -> int: # Ans: 2414
    input = d4_parse(input)
    n = 0
    def find(i, j):
        nonlocal n
        start_i, start_j = i, j
        path = "X"
        dirs = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in dirs:
            for _ in range(3):
                i += dr
                j += dc
                path += input[r][c]
            if path == "XMAS": n += 1
            path, i, j = "X", start_i, start_j

    for r in range(len(input)):
        for c in range(len(input[0])):
            if input[r][c] == "X": find(r, c)
    return n

def day4_solve2(input: str) -> int: # Ans: 1871
    input = d4_parse(input)
    n = 0
    matches = {"M":"S","S":"M"}
    def find_x(i, j):
        nonlocal n
        if (i+1 >= len(input) or j+1 >= len(input[0]) or i==0 or j==0): return
        check = (input[i+1][j+1],input[i-1][j-1],input[i+1][j-1],input[i-1][j+1])
        if (any(c not in matches for c in check) 
            or matches[check[0]] != check[1]
            or matches[check[2]] != check[3]): return
        n += 1
    for r in range(len(input)):
        for c in range(len(input[0])):
            if input[r][c] == "A": find_x(r, c)
    return n
