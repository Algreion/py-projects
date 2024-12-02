# https://adventofcode.com/2024/day/1
import time

def d1_distance_solve(arr):
    arr1, arr2 = arr
    diff = 0
    res = zip(sorted(arr1),sorted(arr2))
    for i in res:
        diff += abs(i[0]-i[1])
    return diff

def d1_separate(string):
    arr1 = []
    arr2 = []
    for i,n in enumerate(string.split()):
        if not i%2: arr1.append(int(n))
        else: arr2.append(int(n))
    return [arr1,arr2]

def day1_solve1(input: str) -> int: # Ans: 3714264
    return d1_distance_solve(d1_separate(input))

def day1_solve2(input: str) -> int: # Ans: 18805872
    start = time.perf_counter()
    n = 0
    arr1, arr2 = d1_separate(input)
    hist = {}
    for v in arr2:
        hist[v] = hist.get(v, 0) + 1
    for v in arr1:
        n += v*hist.get(v,0)
    print (time.perf_counter()-start)
    return n

def d2_report_split(string):
    x = string.split("\n")
    for i in range(len(x)):
        x[i] = x[i].split()
        for j,n in enumerate(x[i]):
            x[i][j] = int(n)
    return x

def day2_solve1(input: str, split=True) -> int: # Ans: 279
    if split: input = d2_report_split(input)
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
    input = d2_report_split(input)
    for report in input:
        if day2_solve1([report], False):
            n += 1
            continue
        for i in range(len(report)):
            if day2_solve1([report[:i]+report[i+1:]], False):
                n += 1
                break
    return n
