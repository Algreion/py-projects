# https://adventofcode.com/2024
import time
from collections import defaultdict, deque
from itertools import combinations
import heapq
from functools import cache

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

def d10_find_paths(grid: list, r: int, c: int, p2=False) -> int:
    dirs = [(0,1),(1,0),(-1,0),(0,-1)]
    seen = set()
    def search(i: int, j: int, path: int, paths=0):
        if not (0<=i<len(grid)) or not (0<=j<len(grid[0])) or grid[i][j] != path: return 0
        if path == 9:
            if p2: return 1
            if (i,j) in seen: return 0
            seen.add((i,j))
            return 1
        for dr, dc in dirs:
            paths += search(i+dr, j+dc, path+1)
        return paths
    return search(r, c, 0)

def d10_parse(input: str) -> list:
    return [list(map(int, list(line))) for line in input.splitlines()]

def day10_solve(input: str, part2=False) -> int: # Ans1: 531 | Ans2: 1210
    grid = d10_parse(input)
    paths = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0: paths += d10_find_paths(grid, r, c, part2)
    return paths

def d11_blink(stones: list, blinks) -> list:
    def blink(s, depth=0):
        key = (s, depth)
        if depth >= blinks: return 1
        if key in values:
            return values[key]
        if s == 0:
            score = blink(1, depth+1)
        elif not len(str(s)) % 2 and s>9:
            s = str(s)
            s1 = int(s[:len(s)//2])
            s2 = int(s[len(s)//2:])
            score = blink(s1, depth+1) + blink(s2, depth+1)
        else: 
            score = blink(s*2024, depth+1)
        values[key] = score
        return score
    total_score = 0
    values = {}
    for s in stones:
        total_score += blink(s)
    return total_score

def day11_solve(input: str, blinks: int = 25) -> int: # Ans1: 186424 | Ans2: 219838428124832
    stones = list(map(int, input.strip("\n").split(" ")))
    return d11_blink(stones, blinks)


def d12_parse(input: str) -> list:
    return [list(line) for line in input.splitlines()]

def d12_convert(grid: list) -> list:
    areas = set()
    done = set()
    rows, cols = len(grid), len(grid[0])

    def convert(r, c):
        nonlocal done, grid
        area = set()
        letter = grid[r][c]
        dirs = [(0,1),(1,0),(-1,0),(0,-1)]
        def dfs(i, j):
            nonlocal area
            if not (0 <= i < rows) or not (0 <= j < cols) or grid[i][j] != letter or (i,j) in area: return
            area.add((i, j))
            for dr,dc in dirs: 
                dfs(i+dr,j+dc)

        dfs(r,c)
        while letter in areas:
            letter = chr(ord(letter)+1)
        
        for x,y in area:
            grid[x][y] = letter
            done.add((x,y))
    
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in done: convert(r, c)
            areas.add(grid[r][c])
    return grid

def d12_count_corners(grid, i, j):

    def get_area(grid, i, j, c):
        area = set()
        check = [(i, j)]
        while check:
            x, y = check.pop()
            if (x, y) not in area and 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == c:
                area.add((x, y))
                check.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
        return area

    def find_corners(area):
        corners, double = set(), 0
        xs, ys = set(x for x, _ in area), set(y for _, y in area) # Unique x, y | Boundary is min(x/y), max(x/y)
        for y in range(min(ys), max(ys) + 2):
            for x in range(min(xs), max(xs) + 2): # Expand boundary by 1
                index = sum(((x + dx, y + dy) in area) * val
                            for dx, dy, val in [(-1, -1, 1), (-1, 0, 2), (0, -1, 4), (0, 0, 8)])
                if index not in [0, 3, 5, 10, 12, 15]: # Corners are all except these
                    corners.add((x, y))
                if index in [6, 9]:
                    double += 1
        return len(corners) + double

    area = get_area(grid, i, j, grid[i][j])
    return find_corners(area)


def day12_solve(input: str, part2=False) -> int: # Ans1: 1518548 | Ans2: 909564
    grid = d12_convert(d12_parse(input))
    a, p = {}, {}
    dirs = [(0,1),(1,0),(-1,0),(0,-1)]
    rows, cols = len(grid), len(grid[0])
    done = set()
    for r in range(rows):
        for c in range(cols):
            letter = grid[r][c]
            a[letter] = a.get(letter, 0) + 1

            if part2 and letter not in done: # Part 2 perimeter counting
                p[letter] = d12_count_corners(grid, r, c)
                done.add(letter)
            
            for dr, dc in dirs:
                if not (0 <= r+dr < rows) or not (0 <= c+dc < cols) or grid[r+dr][c+dc] != letter:
                    if not part2: p[letter] = p.get(letter, 0) + 1 # Part 1 perimeter counting
    res = 0
    for k in a:
        res += a[k] * p[k]
    return res

def d13_parse(input: str) -> dict:
    res = defaultdict(list)
    i = 0
    for line in input.splitlines():
        if not line: 
            i += 1
            continue
        line = line.replace("+"," ").replace(","," ").replace("="," ")
        for n in line.split():
            if n.isdigit(): res[i].append(int(n))
    return res

def d13_z3_a(n: list, file: str = "day_13-1.txt") -> None:
    """Writes the assertion for one claw machine"""
    with open(file,"a") as f:
        i,ax,ay,bx,by,x,y = n[-1],n[0],n[1],n[2],n[3],n[4],n[5]
        f.write(
f"""
(push)
(declare-const Tokens{i} Int)
(declare-const A{i} Int)
(declare-const B{i} Int)
(declare-const x{i} Int)
(declare-const y{i} Int)

(assert (and (= x{i} {x}) (= y{i} {y})))
(assert (= Tokens{i} (+ B{i} (* 3 A{i}))))
(assert (= x{i} (+ (* B{i} {bx}) (* A{i} {ax}))))
(assert (= y{i} (+ (* B{i} {by}) (* A{i} {ay}))))

(minimize Tokens{i})
(check-sat)
(get-value (Tokens{i}))
(pop)
\n""")

def d13_z3_b(n: list, file: str = "day_13-2.txt") -> None:
    """Writes the assertion for one claw machine"""
    with open(file,"a") as f:
        i,ax,ay,bx,by,x,y = n[-1],n[0],n[1],n[2],n[3],n[4],n[5]
        f.write(
    f"""
(push)
(declare-const Tokens{i} Int)
(declare-const A{i} Int)
(declare-const B{i} Int)
(declare-const x{i} Int)
(declare-const y{i} Int)

(assert (and (= x{i} {x+10000000000000}) (= y{i} {y+10000000000000})))
(assert (= Tokens{i} (+ B{i} (* 3 A{i}))))
(assert (= x{i} (+ (* B{i} {bx}) (* A{i} {ax}))))
(assert (= y{i} (+ (* B{i} {by}) (* A{i} {ay}))))

(minimize Tokens{i})
(check-sat)
(get-value (Tokens{i}))
(pop)
    \n""")

def d13_write(input: str) -> bool:
    """Writes the full Z3 file"""
    nums = d13_parse(input)
    for i in nums:
        nums[i].append(i)
        d13_z3_a(nums[i])
        d13_z3_b(nums[i])
    return True

def d13_solve(result: str) -> int:
    """Reads the result from Z3"""
    tokens = 0
    for line in result.splitlines():
        if len(line) > 33: continue
        for n in line.strip("))").split():
            if n.isdigit(): tokens += int(n)
    return tokens

# Instructions for day 13: 
# 1. Run d13_write(input) for the raw puzzle input.
# 2. Run the day_13-1.txt and day_13-2.txt files with Z3
# 3. Copy paste the entire result of one of the two parts in d13_solve(result)

# Answer 1: 37680
# Answer 2: 87550094242995

# https://adventofcode.com/2024/day/14

def d14_parse(input: str) -> list:
    res = []
    for line in input.splitlines():
        if not line: continue
        robot = []
        line = line.replace("="," ").replace(","," ")
        for c in line.split():
            if c.isdigit() or c.startswith("-"): robot.append(int(c))
        res.append(robot)
    return res

def d14_move(robot: list, time: int = 100) -> tuple:
    x, y, vx, vy = robot
    bx, by = 101, 103
    while time > 0:
        x += vx
        y += vy
        while x >= bx: x -= bx
        while y >= by: y -= by
        while x < 0: x += bx
        while y < 0: y += by
        time -= 1
    if not (0<=x<bx) or not (0<=y<by): print(x,y, robot)
    return (x, y)

def d14_quadrants(grid: list) -> int:
    by, bx = len(grid), len(grid[0])
    q1,q2,q3,q4 = 0, 0, 0, 0
    for y in range(by):
        for x in range(bx):
            n = grid[y][x]
            if n==0: continue
            if y < by//2 and x < bx//2: q1 += n
            elif y < by//2 and x > bx//2: q2 += n
            elif y > by//2 and x < bx//2: q3 += n
            elif y > by//2 and x > bx//2: q4 += n
    return q1*q2*q3*q4

def day14_solve1 (input: str) -> int: # Ans: 228690000
    grid = [[0 for _ in range(101)] for _ in range(103)]
    robots = d14_parse(input)
    for r in robots:
        x,y = d14_move(r)
        grid[y][x] += 1
    return d14_quadrants(grid)

def d14_surely_there_is_a_faster_method(input: str, r: tuple) -> None:
    i, j = r
    for sec in range(i,j):
        print(sec)
        day14_solve2(input, sec)

def day14_solve2 (input: str, s: int = 7093, res=False) -> None: # Ans: 7093
    robots = d14_parse(input)
    grid = [[" " for _ in range(101)] for _ in range(103)]
    for r in robots:
        x,y = d14_move(r, s)
        grid[y][x] = "X"
    if not res: print(grid)
    else: return grid

def day14_visualizer(input: str) -> bool: # Cool tree I guess...
    grid = day14_solve2(input, res=True)
    with open("day_14-tree.txt","w") as f:
        for line in grid:
            f.write("".join(line))
            f.write("\n")
    return True


def d15_push(map: list, r: int, c: int, dr: int, dc: int) -> list:
    if not (0<=r<len(map)) or not (0<=c<len(map[0])) or map[r][c] in "#.": return []
    char, map[r][c] = map[r][c], "."
    comp = [(r, c, char)]
    comp.extend(d15_push(map, r+dr, c+dc, dr, dc))
    if char == "[": comp.extend(d15_push(map, r, c+1, dr, dc))
    if char == "]": comp.extend(d15_push(map, r, c-1, dr, dc))
    return comp

def d15_p2(ch):
  if ch == "@": return ["@", "."]
  elif ch == ".": return [".", "."]
  elif ch == "#": return ["#", "#"]
  else: return ["[", "]"]

def d15_solver(grid: list, moves: str) -> int:
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "@": r, c = i,j
    for move in moves:
        if move == "v": dr, dc = 1, 0
        elif move == "^": dr, dc = -1, 0
        elif move == ">": dr, dc = 0, 1
        elif move == "<": dr, dc = 0, -1
        else: continue
        comp = d15_push(grid, r, c, dr, dc)
        if all(grid[i+dr][j+dc] == "." for i, j, _ in comp):
            comp = [(i+dr, j+dc, ch) for i, j, ch in comp]
            r, c = r+dr, c+dc
        for rr, cc, char in comp:
            grid[rr][cc] = char
    return sum(sum(i*100 + j for j,ch in enumerate(line) if ch in "O[") for i, line in enumerate(grid))

def day15_solve(input: str) -> None: # Ans1: 1371036 | Ans2: 1392847
    map, moves = input.split("\n\n")
    map1 = [list(line) for line in map.split("\n")]
    map2 = [sum((d15_p2(c) for c in line), []) for line in map.split("\n")]
    print(f"Part 1 solution: {d15_solver(map1, moves)}")
    print(f"Part 2 solution: {d15_solver(map2, moves)}")

import heapq

def d16_parse(input: str) -> tuple:
    maze = [list(line) for line in input.splitlines()]
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == "S": start = (r,c)
            if maze[r][c] == "E": end = (r,c)
    return (maze,start,end)

def d16_pathfind(grid: list, start: tuple, end: tuple) -> int:
    check = [(0, *start, (0, 0))]
    processed = set()
    dirs = [(0,1),(1,0),(-1,0),(0,-1)]
    rows, cols = len(grid), len(grid[0])
    while check:
        s, x, y, d = heapq.heappop(check)
        if (x,y) == end: return s
        processed.add((x,y))
        for dr,dc in dirs:
            r,c = x+dr,y+dc
            if not (0<=r<rows) or not (0<=c<cols) or grid[r][c] == "#": continue
            elif (r,c) not in processed:
                score = s+1 if (dr,dc) == d else s+1001
                heapq.heappush(check, (score, r,c, (dr, dc)))
    return False

def day16_solve(input: str) -> int: # Ans: 66404
    info = d16_parse(input)
    return d16_pathfind(*info)

import heapq

D18_ROWS, D18_COLS = 71, 71

def d18_parse(input: str, ns: int = 1024) -> list:
    corrupt = [tuple(map(int,line.split(","))) for line in input.splitlines()[:ns]]
    grid = [["." for _ in range(D18_COLS)] for _ in range(D18_ROWS)]
    for x,y in corrupt:
        grid[x][y] = "#"
    return grid

def d18_distance(x,y):
    xe,ye = D18_COLS-1, D18_COLS-1
    return abs(x-xe) + abs(y-ye) # (Manhattan Distance)

def d18_pathfind(grid: list) -> int:
    start, end = (0,0), (D18_COLS-1, D18_COLS-1)
    check = [(0, *start, 0)]
    processed = set()
    dirs = [(0,1),(1,0),(-1,0),(0,-1)]
    best = {start: 0}
    while check:
        _, x, y, s = heapq.heappop(check)
        if (x,y) == end: return s
        if (x,y) in processed: continue
        processed.add((x,y))
        for dr,dc in dirs:
            r,c = x+dr,y+dc
            if not (0 <= r < D18_ROWS) or not (0 <= c < D18_COLS) or grid[r][c] == "#" or (r,c) in processed: continue
            score = s+1
            if (r,c) not in best or score < best[(r,c)]:
                best[(r,c)] = score
                dist = score + d18_distance(r,c)
                heapq.heappush(check, (dist, r, c, score))
    return -1


def day18_solve1(input: str) -> int: # Ans: 354
    return d18_pathfind(d18_parse(input))

def day18_solve2(input: str, loading: bool = True) -> int: # Ans: 36,17
    for ns in range(1024, len(input.splitlines())):
        if not ns % 100 and loading: print(ns)
        if d18_pathfind(d18_parse(input, ns)) == -1: return input.splitlines()[ns-1] # -1 because 0-indexed

@cache
def d19_check(pattern: str, towels: list) -> int:
    n = 0
    for t in towels:
        if pattern == t: n += 1
        if pattern.startswith(t): n += d19_check(pattern.removeprefix(t), towels)
    return n

def day19_solve(input: str) -> None: # Ans1: 302 | Ans2: 771745460576799
    total1 = 0
    total2 = 0
    towels, patterns = input.splitlines()[0], input.splitlines()[1:]
    towels = tuple(sorted([t for t in towels.split(", ")], key=len, reverse=True))
    for p in patterns:
        x = d19_check(p, towels)
        total2 += x
        total1 += (x != 0)
    print(f"Part 1 | Number of possible designs: {total1}")
    print(f"Part 2 | Number of unique designs: {total2}")


def d22_pr(n: int) -> int:
    n ^= n<<6
    n %= 1<<24
    n ^= n>>5
    n %= 1<<24
    n ^= n<<11
    n %= 1<<24
    return n

def d22_run(n: int) -> int:
    for _ in range(2000):
        n = d22_pr(n)
    return n

def day22_solve(input: str) -> int: # Ans: 13429191512
    return sum(d22_run(int(n)) for n in input.splitlines())

def d24_solver(input: str, fin: str = "day_24.clj") -> None:
    with open(fin, "w") as f:
        in1, in2 = input.split("\n\n")
        const = set()
        for line in in1.splitlines():
            x,y = line.split(": ")
            const.add(x)
            f.write(f"(declare-const {x} Bool)\n(assert (= {x} {"true" if int(y) else "false"}))\n")
        for line in in2.splitlines():
            a, b, c, d = line.replace("-> ","").split()
            for x in [a,c,d]:
                if x not in const: 
                    f.write(f"\n(declare-const {x} Bool)")
                    const.add(x)
            f.write(f"\n(assert (= ({b.lower()} {a} {c}) {d}))")
        f.write("\n(check-sat)\n(get-value (")
        for x in sorted(const, reverse=True):
            if x[0] == "z":
                f.write(f"{x} ")
            else:
                f.write("))")
                break

def day24_solve(result: str) -> int: # Ans: 49430469426918
    res = ""
    for line in result.splitlines():
        if "true" in line: res += "1"
        elif "false" in line: res += "0"
    return int(res, 2)

def d25_parse(input: str) -> tuple:
    keys, locks = {}, {}
    input = [item.splitlines() for item in input.split("\n\n")]
    h = len(input[0])-1
    for item in input:
        lock = "#" in item[0]
        heights = {}
        for i,line in enumerate(item):
            for j, c in enumerate(line):
                if lock and c != "#" and j not in heights: heights[j] = i-1
                elif not lock and c == "#" and j not in heights: heights[j] = h-i
        heights = tuple(v for _,v in sorted(heights.items()))
        group = locks if lock else keys
        group[heights] = group.get(heights, 0) + 1
    return (keys, locks, h)

def day25_solve(input: str) -> int: # Ans: 3663
    keys, locks, h = d25_parse(input)
    matches = 0
    for k, k_i in keys.items():
        for l, l_i in locks.items():
            if all(k[i] + l[i] < h for i in range(len(k))): matches += k_i * l_i

# Merry Christmas!
