import random


def bitflip_mutation(p: list, prob: float | None = None) -> list:
    f = lambda x: int(not(x))
    if prob is None: prob = 1/max(1,len(p))
    for i in range(len(p)):
        if random.random()<=prob: p[i] = f(p[i])
    return p

def swap_mutation(p: list, ind: tuple | None = None) -> list:
    if ind is None: ind = (random.randint(0,max(0,len(p)-1)),random.randint(0,max(0,len(p)-1)))
    p[ind[0]],p[ind[1]] = p[ind[1]],p[ind[0]]
    return p

def crossover2(p1: list, p2: list, cut: tuple | None = None) -> tuple:
    if cut is None: cut = (random.randint(0,len(p1)),random.randint(0,len(p1)))
    return p1[:cut[0]]+p2[cut[0]:cut[1]]+p1[cut[1]:],p2[:cut[0]]+p1[cut[0]:cut[1]]+p2[cut[1]:]

def crossover1(p1: list, p2: list, cut: int | None = None) -> tuple:
    if cut is None: cut = random.randint(0,len(p1))
    return p1[:cut]+p2[cut:],p2[:cut]+p1[cut:]

def odx1(p1: list, p2: list, cut: tuple | None = None) -> tuple:
    if cut is None: cut = (random.randint(0,len(p1)),random.randint(0,len(p1)))
    cut1,cut2 = p1[cut[0]:cut[1]],p2[cut[0]:cut[1]]
    P1 = [i for i in p2[cut[1]:]+p2[:cut[1]] if i not in set(cut1)]
    P2 = [i for i in p1[cut[1]:]+p1[:cut[1]] if i not in set(cut2)]
    l = len(p1)-cut[1]
    return P1[l:]+cut1+P1[:l],P2[l:]+cut2+P2[:l]

def tournament_selection(pop: list, fit: list, k: int) -> list:
    p = []
    ind = [i for i in range(len(pop))]
    for _ in range(k):
        p1,p2 = random.sample(ind,2)
        f1,f2 = fit[p1],fit[p2]
        if f1 > f2: p.append(pop[p1])
        elif f1 < f2: p.append(pop[p2])
        else: p.append(random.choice([pop[p1],pop[p2]]))
    return p

def roulette_selection(pop: list, fit: list, k: int) -> list:
    fitsum = sum(fit)
    probs = [f/fitsum for f in fit]
    return random.choices(pop,weights=probs,k=k)

def sort_ga(pop: tuple | list = (4,6), gen: int = 100) -> list:
    if isinstance(pop,tuple): pop = [random.sample([i for i in range(1,pop[1]+1)],pop[1]) for _ in range(pop[0])]
    sort_fit = lambda c: sum([int(c[i-1]<c[i]) if i>0 else 0 for i in range(len(c))])
    found = False
    for g in range(gen):
        fit = [sort_fit(p) for p in pop]
        best = max(fit)
        if best == len(pop[0])-1:
            print(f"Found a perfect sort with fitness {best} after {g+1} generations!")
            found = True
            break
        parents = tournament_selection(pop,fit,len(pop))
        offspring = [odx1(parents[i],parents[i+1]) for i in range(len(parents)//2)]
        offspring = [o[0] for o in offspring]+[o[1] for o in offspring]
        pop = [swap_mutation(o) for o in offspring]
    if not found:
        fit = [sort_fit(p) for p in pop]
        print(f"Best fitness after {gen} generations: {max(fit)}.")
    return pop[fit.index(best)]
