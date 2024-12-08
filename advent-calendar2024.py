# https://adventofcode.com/2024
import time
from collections import defaultdict, deque
from itertools import combinations
import heapq

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

def d5_parse(input: str) -> dict:
    # Parsing instructions
    input = input.split("\n")
    for i in range(len(input)):
        input[i] = input[i].split("|")
        input[i][0], input[i][1] = int(input[i][0]), int(input[i][1])
    # Initializing dictionary
    nodes = {}
    for pair in input:
        if pair[0] not in nodes:
            nodes[pair[0]] = set()
        if pair[1] not in nodes:
            nodes[pair[1]] = set()
        nodes[pair[0]].add(pair[1])
    return nodes
def d5_parse_input(input: str) -> list:
    input = input.split("\n")
    for i in range(len(input)):
        input[i] = input[i].split(",")
        for j in range(len(input[i])): input[i][j] = int(input[i][j])
    return input
def d5_check(nodes: dict, pages: list) -> bool:
        seen = set()
        for v in pages:
            for n in nodes.get(v, set()):
                if n in seen: return False
            seen.add(v)
        return True
def d5_reorder(dependencies: dict, pages: list):
    if d5_check(dependencies, pages): return False
    # Doubly linked graph of dependencies
    positions = {}
    for key, after in dependencies.items():
        if key not in positions:
            positions[key] = {"after": set(), "before": set()}
        positions[key]["after"].update(after)

        for after_i in after:
            if after_i not in positions:
                positions[after_i] = {"after": set(), "before": set()}
            positions[after_i]["before"].add(key)
    
    # Topological sorting
    def find_min_index(item, path):
        if item not in positions:
            return len(path)
        
        after_min_index = 0
        for after_i in positions[item].get("after", set()):
            if after_i in path:
                after_min_index = max(after_min_index, path.index(after_i) + 1)
        
        before_max_index = len(path)
        for before_i in positions[item].get("before", set()):
            if before_i in path:
                before_max_index = min(before_max_index, path.index(before_i))

        return min(after_min_index, before_max_index)
    
    sorted_list = []
    remaining = set(pages)
    
    while remaining:
        for item in list(remaining):
            index = find_min_index(item, sorted_list)

            sorted_list.insert(index, item)
            remaining.remove(item)
            break
    
    return sorted_list[::-1]

def day5_solve1(instructions: str, input: str) -> int: # Ans: 4662
    nodes = d5_parse(instructions)
    input = d5_parse_input(input)
    # Solve
    res = 0
    for pages in input:
        if d5_check(nodes, pages):
            res += pages[len(pages)//2]
    return res

def day5_solve2(instructions: str, input: str) -> int: # Ans: 5900
    nodes = d5_parse(instructions)
    input = d5_parse_input(input)
    res = 0
    for pages in input:
        pages = d5_reorder(nodes, pages)
        if pages:
            res += pages[len(pages)//2]
    return res

def d6_parse(input: str) -> list:
    input = input.split("\n")
    for i in range(len(input)):
        input[i] = list(input[i])
    return input

def day6_solve1(input: str) -> int: # Ans: 5153
    grid = d6_parse(input)
    rows, cols = len(grid),len(grid[0])
    guard = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in "<v^>": 
                guard = (r,c)
                break
        if guard: break
    directions = {"^":((-1,0),">"),"v":((1,0),"<"),"<":((0,-1),"^"),">":((0,1),"v")}
    i,j = guard
    while 0<=i<rows and 0<=j<cols:
        guard = grid[i][j]
        dr,dc = directions[guard][0]
        if 0<=i+dr<rows and 0<=j+dc<cols:
            if grid[i+dr][j+dc] == "#":
                grid[i][j] = directions[guard][1]
                continue
            grid[i+dr][j+dc] = guard
        grid[i][j] = "X"
        i += dr
        j += dc
    path = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "X": path += 1
    return path

def d6_move(grid, start, rows, cols):
    directions = {"^": ((-1, 0), ">"), "v": ((1, 0), "<"), "<": ((0, -1), "^"), ">": ((0, 1), "v")}
    visited = set()
    i, j, direction = start

    while 0 <= i < rows and 0 <= j < cols:
        cell = (i, j, direction)
        if cell in visited: return True # Loop check
        visited.add(cell)

        dr, dc = directions[direction][0]
        if 0 <= i + dr < rows and 0 <= j + dc < cols and grid[i + dr][j + dc] == "#":
            direction = directions[direction][1]
        else:
            i += dr
            j += dc

    return False 

def day6_solve2(input: str, load=True) -> int: # Ans: 1711
    grid = d6_parse(input)
    rows, cols = len(grid), len(grid[0])
    start = None

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in "<v^>":
                start = (r, c, grid[r][c])
                grid[r][c] = "."
                break

    paradox = 0

    for r in range(rows):
        for c in range(cols):
            if load: print(r,c)
            if grid[r][c] == ".":
                grid[r][c] = "#"
                if d6_move(grid, start, rows, cols):
                    paradox += 1
                grid[r][c] = "." # Backtrack

    return paradox

def d7_parse(input: str) -> list:
    return [list(map(int, line.replace(":","").split()))
            for line in input.splitlines()]
            
def d7_check(target: int, vals: list, i: int=0, path: int=0, two=False) -> bool:
    if i == 0:
        return d7_check(target, vals, 1, vals[0], two)
    else:
        if i >= len(vals):
            return True if target == path else False
        if two:
            return (d7_check(target, vals, i+1, path+vals[i], True) or d7_check(target, vals, i+1, path*vals[i], True)
                    or d7_check(target, vals, i+1, int(str(path)+str(vals[i])), True))
        return d7_check(target, vals, i+1, path+vals[i]) or d7_check(target, vals, i+1, path*vals[i])

def day7_solve(input: str, part2=False) -> int: # Ans1: 1399219271639 | Ans2: 275791737999003
    input = d7_parse(input)
    res = 0
    for item in input:
        if d7_check(item[0], item[1:], two=part2): res += item[0]
    return res

def d8_parse(input: str) -> list:
    return [list(row) for row in input.split("\n")]

def d8_antinodes1(grid: list, antennas: dict) -> int:
    antinodes = set()
    rows, cols = len(grid), len(grid[0])
    for key in antennas:
        for (r1, c1),(r2,c2) in combinations(antennas[key],2):
            dr, dc = r2 - r1, c2-c1
            if 0 <= r1-dr < rows and 0 <= c1-dc < cols: antinodes.add((r1-dr, c1-dc))
            if 0 <= r2+dr < rows and 0 <= c2+dc < cols: antinodes.add((r2+dr, c2+dc))
    return len(antinodes)

def d8_antinodes2(grid: list, antennas: dict) -> int:
    antinodes = set()
    rows, cols = len(grid), len(grid[0])
    for key in antennas:
        for (r1, c1),(r2, c2) in combinations(antennas[key],2):
            dr, dc = r1-r2, c1-c2
            for i in range(cols):
                ax, ay = r1 + i*dr, c1 + i*dc
                bx, by = r2 - i*dr, c2 - i*dc
                if 0 <= ax < rows and 0 <= ay < cols: antinodes.add((ax, ay))
                if 0 <= bx < rows and 0 <= by < cols: antinodes.add((bx, by))
    return len(antinodes)

def day8_solve(input: str, part2=False) -> int: # Ans1: 426 | Ans2: 1359
    grid = d8_parse(input)
    frequencies = defaultdict(set)
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != ".": frequencies[grid[r][c]].add((r,c))
    if part2: return d8_antinodes2(grid, frequencies)
    return d8_antinodes1(grid, frequencies)

def day9_solve1(input: str) -> int: # Ans: 6385338159127
    blocks = deque(enumerate(int(n) for n in input[::2]))
    space = deque(int(n) for n in input[1::2])
    res = []
    while blocks:
        new_block = blocks.popleft()
        res.append(new_block)
        if space:
            new_gap = space.popleft()
            while blocks and new_gap:
                id, size = blocks.pop()
                if size <= new_gap:
                    res.append((id, size))
                    new_gap -= size
                else:
                    res.append((id, new_gap))
                    blocks.append((id, size-new_gap))
                    new_gap = 0
    checksum = 0
    i = 0
    for id, size in res:
        checksum += id*size*(2*i + size-1)//2
        i += size
    return checksum

def day9_solve2(input: str) -> int: # Ans: 6415163624282
    is_block = True
    blocks = []
    gaps = [[] for _ in range(10)]
    pos, id = 0, 0
    for d in input:
        d = int(d)
        if is_block:
            blocks.append([pos, id, d])
            id += 1
        else:
            heapq.heappush(gaps[d], pos)
        pos += d
        is_block = not is_block
    checksum = 0
    for b in range(len(blocks))[::-1]:
        pos, id, size = blocks[b]
        best = pos
        candidates = [(heapq.heappop(gaps[i]), i) for i in range(10) if gaps[i] and i>=size]
        if candidates:
            gpos, glen = min(candidates)
            if gpos < pos:
                best = gpos
                candidates.remove((gpos, glen))
                heapq.heappush(gaps[glen-size], gpos+size)
            for gpos, glen in candidates:
                heapq.heappush(gaps[glen], gpos)
            blocks[b][0] = best
        checksum += id*size*(2*best + size-1)//2
    return checksum
