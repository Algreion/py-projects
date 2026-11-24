# Python Projects
Main repo for all the Python projects that don't neatly fit into any other repository.

## Fraction
'fraction.py' is a custom library that implements a Fraction object without using division at all, able to store rational numbers and perform operations on them.

**Requirements:** N/A

**Usage:**
```python
from fraction import Frac

a = Frac(10,3) # 3.333...
b = Frac(0.5) # 1/2

c = a*b
print(c) # 5/3
print(float(c) == 5/3) # True
```

## Chess
'chess.py' is a custom CLI chess UI, and can be played against other players, a bot or an API.

The project was ported to a terminal version, while the original pygame version is available in [pygame-playground](https://github.com/Algreion/pygame-playground). Type 'help' to see all options.

**Requirements:**

`pip install rich requests`

**Usage:**

`python chess.py`

## rando
A custom pseudo-random generator library using a logistic map. Will likely be moven to a separate repo after adding more features, and a suite of statistical testing to prove its effectiveness.

**Requirements:** N/A

## alglib
A library of useful classes and helper functions implemented from scratch:
- Collections: flatten, rev_dict, remove_duplicates, powerset, subsets
- Strings: snake_to_camel, camel_to_snake, is_anagram, is_palindrome, valid_parenthesis, checkp
- Numbers: is_prime, prime_list, byte_converter, crgb, chex, invert
- Text analysis: common_letters, common_chars, common_words
- Classes: MinHeap, MaxHeap, Hashset, Hashmap

**Requirements:** None

## alglib_exp
A library of various experimental functions and classes made over the years. Fun, but less useful. A few examples include: to_morse and from_morse, solveNQueens, weekday, timer, LinkedList and Keyboard.

**Requirements:** None

## Wordle
A WIP project for a CLI solver which will be moved to its own repository when it is completed.

## Neural Networks
Small projects involving neural networks that will soon be moved to their own repo:
- digits: Standard MNIST digit recognition
- tokenizer: Custom BPE tokenizer made from scratch
- ngram and bigrams: PyTorch projects to generate new English names.

## Early Code
Many of my earliest coding projects and experiments with Python. Most have little use, but I decided to keep them for sentimental value.