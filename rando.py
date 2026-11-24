# Custom pseudo-random number geneator from logistic map.
import time,math

CHAOS = 4.0 # 4 is true chaos but unstable
MAXSEED = 2**30

def f(x: float,r: float):
    """Logistic Map. x >= 0, 0 <= r <= 4"""
    return r*x*(1-x)

def I(x: float, r: float, i: int = 100, func=f):
    """Iterations on function."""
    for _ in range(i):
        x = func(x,r)
    return x

def random(seed: float = None) -> float:
    """Returns a random float between 0 and 1."""
    if seed is None: seed = time.time()
    seed = seed % MAXSEED
    seed /= MAXSEED
    x = max(1e-12, min(1-1e-12, I(seed,CHAOS,100)))
    return (2/math.pi)*math.asin(math.sqrt(x)) # Normalize to uniform distribution

def randint(a: int, b: int, seed: float = None) -> int:
    """Returns a random integer in [a,b]"""
    return int(random(seed)*(b-a+1)+a)

def quick_test(a: int, b: int, n: int = 100000):
    """Test probability distribution. Better statistics tests coming soon."""
    d = dict()
    for _ in range(n):
        r = randint(a,b)
        d[r] = d.get(r,0)+1
    for k in d:
        print(f"{k}: {100*d[k]/n:.2f}%")
    print(f"Expected Value: {sum(k*v/n for k,v in d.items())}")
