"""Library of old functions I made, with dubious utility."""

import random
import time

# 1. Games & Bots

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

# 2. Functions - Interactive programs

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

def sum_product():
    n = int(input("Insert a positive integer: "))
    if n <= 0 or not isinstance(n,int):
        print("We don't do that here.")
        return
    c = input("""Type "s" for the sum of integers 1 through n, "p" for the product: """)
    if c == "s":
        result = int(((n+1)*n)/2)
        print (result)
    elif c == "p":
        result = 1
        for i in range(n):
            result *= (i+1)
        print (result)
    else:
        print("Follow the instructions properly next time.")
        return

# 3. Functions - Time

def sconverter(x=None):
    minutes = 60
    hours = 60*minutes
    days = 24*hours
    years = 365*days
    if x is None:
        x = int(input("Enter a number of seconds: "))
    s = int(x)
    y = s//years
    if y > 0:
        s = s%(y*years)
    d = s//days
    if d > 0:
        s = s%(d*days)
    h = s//hours
    if h > 0:
        s = s%(h*hours)
    m = s//minutes
    if m > 0:
        s = s%(m*minutes)
    print(f"That is {y:,} years, {d} days, {h} hours, {m} minutes and {s} seconds.")
    if y > 9999999999999: # Funny right?
        print(f"""I extend my heartfelt gratitude for utilizing my humble converter, o esteemed Immortal, 
for that is approximately {y//(10**13):,} chaos cycles!""")
    elif y > 999999999:
        print(f"That is {y//(10**9):,} billion years, I'm sure the precision here is needed...")
    elif y > 999:
        print(f"Wow! That's {y//1000:,} millennia, what are you using this for?")

def converter(x=None,n=None):
    # I know it's the opposite of efficient, but hey at least it works!
    minutes = 60
    hours = 60*minutes
    days = 24*hours
    years = 365*days
    if n is None:
        n = input("What unit do you want to convert? ")
    if n == "s":
        n = "seconds"
    if n == "m":
        n = "minutes"
    if n == "h":
        n = "hours"
    if n == "d":
        n = "days"
    if n == "y":
        n = "years"
    if x is None:
        x = float(input(f"Enter the number of {n}: "))
    if x < 0:
        print("I'm sorry, but negative time hasn't been invented yet!")
        return
    if n == "seconds":
        sconverter(x)
    elif n == "minutes":
        x = x*minutes
        sconverter(x)
    elif n == "hours":
        x = x*hours
        sconverter(x)
    elif n == "days":
        x = x*days
        sconverter(x)
    elif n == "years":
        x = x*years
        sconverter(x)
    else:print("The only valid units of time are s (seconds), m (minutes), h (hours) and d (days).")
    ask = input("Want to convert again (y/n)? ")
    if ask == "y":
        converter()
    else:print("Have a nice day!") # :D

def ywd(d: int = None) -> None:
    """Days to year + week + days"""
    if d == None:
        d = int(input("Insert a number of days: "))
    if d < 0:
        print ("Days must be positive.")
    years,days = divmod(d,365)
    weeks,days = divmod(days,7)
    print (f"{d} days = %d years + %d weeks + %d days" % (years,weeks,days))

def clock() -> str:
    t = time.localtime()
    print(f"Current time: {t[0]}/{t[1]:02d}/{t[2]}, {t[3]:02d}:{t[4]:02d}:{t[5]:02d}.")

# 4. Print & patterns

def fizzbuzzer(n: int) -> None:
    i = 1
    while i <= n:
        print("FizzBuzz" if not i % 15 else "Buzz" if not i % 5 else "Fizz" if not i % 3 else i)
        i += 1

def abc(n:int, string="") -> None:
    if len(string) == n:
        print(string)
    elif len(string) > n:
        return
    else:
        for c in ["A", "B", "C"]:
            abc(n, string + c)

def diagonal_matrix(n: int, x: str = "X", space: str = ".") -> str:
    if n <= 0:
        return None
    return "\n".join(" ".join(x if i == j else space for i in range(n)) for j in range(n))

def x_matrix(n: int, x: str = "X", space: str = ".") -> str:
    if n <= 0:
        return None
    return "\n".join(" ".join(x if i == j or i == n - j - 1 else space for i in range(n)) for j in range(n))

def pyramid(h: int, brick: str = "#", air: str = " ") -> None:
    for i in range(h):
        print(air*(h-i),brick*(1+2*i),air*(h-i))

def pyramid2(h: int = 3, brick: str = "#", air: str = " "):
    for i in range(1,h*2+1,2):
        print((brick*i).center(h*2-1, air))

def cube(l: int, brick: str = "#") -> None:
    for _ in range(l):
        print(brick*l)