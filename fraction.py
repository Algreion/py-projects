

class Fraction:
    def __init__(self, a: int | float | str, b: int | None = None, precision: int = 1000000):
        """Fraction object that doesn't utilize any division at all."""
        self._num = None
        self._den = None
        self._neg = 1
        self._precision = precision
        if b is None:
            if isinstance(a,str):
                self._num, self._den = self._parse_string(a)
            else:
                if a < 0:
                    self._neg *= -1
                    a *= -1
                self._num, self._den = self._closest_fraction(a)
        else:
            self._num = int(a)
            self._den = int(b)
        self.simplify()
    
    def __str__(self):
        return f"{self._num*self._neg}/{self._den}" if self._den != 1 else str(self._num*self._neg)
    def __repr__(self):
        return f"Frac({self._num*self._neg},{self._den})"
    
    def __add__(self, other: any):
        if not isinstance(other,Fraction): other = Fraction(other,precision=self._precision)
        lcm,ma,mb = self._lcm(self._den,other._den,True)
        return Fraction(ma*self._num*self._neg+mb*other._num*other._neg,lcm,precision=self._precision)
    def __radd__(self, other):
        return self+other

    def __neg__(self):
        res = Fraction(self._num,self._den,precision=self._precision)
        res._neg = -1*self._neg
        return res

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, power: int):
        if isinstance(power,Fraction):
            power = power._num if power._den == 1 else power._evaluate()
        if not isinstance(power,int):
            raise TypeError('Only integers accepted for exponentiation.')
        P = power < 0
        power = abs(power)
        num = self._num**power
        den = self._den**power
        if P: num, den = den, num
        return Fraction(num,den,precision=self._precision)
    
    def __mul__(self, other):
        if not isinstance(other,Fraction): other = Fraction(other,precision=self._precision)
        return Fraction(self._num*other._num*self._neg*other._neg,self._den*other._den,precision=self._precision)
    def __rmul__(self, other):
        return self*other
    
    def __truediv__(self, other):
        if not isinstance(other,Fraction): other = Fraction(other,precision=self._precision)
        other **= -1
        return self*other
    def __rtruediv__(self, other):
        return self/other
    
    def simplify(self) -> None:
        if self._den == 0:
            raise ZeroDivisionError
        if self._num == 0:
            self._den = 1
            self._neg = 1
            return
        if self._num<0:
            self._num *= -1
            self._neg *= -1
        if self._den<0:
            self._den *= -1
            self._neg *= -1
        fnum, dnum = self._factors(self._num),self._factors(self._den)
        nnew,dnew = fnum.copy(),dnum.copy()
        for fn in fnum:
            if fn in dnum: nnew[fn] = max(0,fnum[fn]-dnum[fn])
        for dn in dnum:
            if dn in fnum: dnew[dn] = max(0,dnum[dn]-fnum[dn])
        num,den = 1,1
        for k in nnew: num*=k**nnew[k]
        for k in dnew: den*=k**dnew[k]
        self._num,self._den = num,den

    def evaluate(self) -> float:
        return self._num/self._den
    
    def lcm(self, other: any) -> int:
        if not isinstance(other,Fraction): other = Fraction(other,precision=self._precision)
        return self._lcm(self._den,other._den)
    def __matmul__(self,other: any) -> int:
        return self.lcm(other)
    
    def __float__(self) -> float:
        return self.evaluate()
    def __int__(self) -> int:
        return int(self.evaluate())
    def __iter__(self) -> list:
        return iter([self._num, self._den])
    
    def _lcm(self, a: int, b: int, multipliers: bool = False) -> int | tuple:
        a_options, b_options = [], []
        a_multi, b_multi = [],[]
        ab = a*b
        i = 1
        while True:
            if a*i > ab and b*i > ab: break
            a_options.append(a*i)
            b_options.append(b*i)
            i += 1
        res = min(filter(lambda x: x in b_options,a_options))
        if multipliers: return (res, a_options.index(res)+1, b_options.index(res)+1)
        return res

    def _parse_string(self, string: str) -> tuple:
        try:
            a,b = [int(i) for i in string.split('/')]
            return a,b
        except:
            return tuple()

    def _factors(self, n: int) -> dict:
        factors = {}
        partial = 1
        if not isinstance(n,int) or n<2: return factors
        for i in range(2,n+1):
            if n%i or any(i%x==0 for x in factors): continue
            x = 1
            while True:
                if n%i**(x+1): break
                x+=1
            factors[i] = x
            partial *= i**x
            if partial == n: break
        return factors

    def _closest_fraction(self, n: float, max_a: int = None, max_b: int = None) -> tuple:
        if max_a is None: max_a = self._precision
        if max_b is None: max_b = self._precision
        best_a = best_b = 0
        best_err = float("inf")
        for b in range(1,max_b+1):
            a = round(n*b)
            if a<0 or a>max_a:
                continue
            err = abs(a/b-n)
            if err < best_err:
                best_err = err
                best_a, best_b = a,b
            if best_err == 0: break
        return best_a,best_b
