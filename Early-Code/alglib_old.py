"""Library of old classes and functions I made, with dubious utility."""

import random

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