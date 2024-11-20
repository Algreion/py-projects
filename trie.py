from collections import deque
import string
from difflib import get_close_matches as gcm

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            c = c.lower()
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.end = True

    def normal_search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children: return False
            cur = cur.children[c]
        return cur.end
    
    def search(self, word: str) -> bool:
        """Searches for a word with wildcards using ".", such as "g..d" -> True if good, goad etc. are present."""
        q = deque([(self.root,0)])
        while q:
            node, depth = q.popleft()
            if depth == len(word) and node.end: return True
            if depth == len(word) and not node.end: continue
            c = word[depth]
            if c != ".":
                if c not in node.children: continue
                q.append((node.children[c],depth+1))
            else:
                for child in node.children:
                    q.append((node.children[child],depth+1))
        return False
    
    def complete(self, word: str) -> str:
        """Fills in the wildcards in alphabetical order, such as "a..le" -> "abele"."""
        q = deque([(self.root, "", 0)])
        while q:
            node, path, depth = q.popleft()
            if depth == len(word):
                if node.end: return path
                continue
            c = word[depth]
            if c != ".":
                if c in node.children: q.append((node.children[c], path+c, depth+1))
            else:
                for char in sorted(node.children):
                    q.append((node.children[char], path + char, depth+1))
        return ""
    
    def autocomplete(self, prefix: str) -> str:
        """Same as normal autocomplete but accepts wildcards."""
        q = deque([(self.root, "", 0)])
        results = []
        while q:
            node, path, index = q.popleft()
            if index < len(prefix):
                c = prefix[index]
                if c != ".":
                    if c in node.children:
                        q.append((node.children[c], path + c, index + 1))
                else:
                    for char in sorted(node.children):
                        q.append((node.children[char], path + char, index + 1))
            else:
                if node.end: results.append(path)
                for c in sorted(node.children):
                    q.append((node.children[c], path+c, index+1))
        return ", ".join(results)

    def wordlist(self, prefix: str) -> list:
        """Returns a list of words that match the wildcard+letter, such as "...le"."""
        q = deque([(self.root, "", 0)])
        words = []
        while q:
            node, path, depth = q.popleft()
            if depth == len(prefix):
                if node.end: words.append(path)
                continue
            c = prefix[depth]
            if c != ".":
                if c in node.children: q.append((node.children[c], path+c, depth+1))
            else:
                for char in sorted(node.children):
                    q.append((node.children[char], path + char, depth+1))
        return words

    def startswith(self, prefix: str) -> bool:
        """Returns True if there's any word starting with the string provided, else False."""
        cur = self.root
        for c in prefix:
            if c not in cur.children: return False
            cur = cur.children[c]
        return True
    
    def insertmulti(self, words: list) -> None:
        for w in words:
            self.insert(w)
    
    def get_depth(self) -> int:
        """Returns the depth of the Trie."""
        def depth(node):
            if not node.children: return 0
            return 1+max(depth(child) for child in node.children.values())
        return depth(self.root)
    
    def normal_autocomplete(self,prefix: str) -> str:
        """Returns the first word in alphabetical order that begins with the prefix, or "" if none match."""
        cur = self.root
        for c in prefix:
            if c not in cur.children: return ""
            cur = cur.children[c]
        q = deque([(cur,prefix)])
        while q:
            node, path = q.popleft()
            if node.end: return path
            for c in sorted(node.children):
                q.append((node.children[c], path + c))
        return ""
    
    def fill(self,file: str) -> None:
        """Fills out the Trie with a txt file's words."""
        punctuation = str.maketrans("","",string.punctuation)
        with open(file,"r",encoding="utf-8") as f:
            for line in f:
                for word in line.split():
                    self.insert(str(word.translate(punctuation)))

    def normal_wordlist(self, n: int=None) -> list:
        """Returns the n first words in the Trie in alphabetical order."""
        def collect_words(node,prefix,words,n):
            if node.end: words.append(prefix)
            if n is not None and len(words) >= n: return
            for c in sorted(node.children):
                if n is not None and len(words) >= n: return
                collect_words(node.children[c], prefix+c,words,n)
        words = []
        collect_words(self.root,"",words,n)
        return words
    
    def delete(self, word: str) -> bool:
        """Removes a single word from the Trie."""
        present = False
        def deletor(node, index):
            nonlocal present
            if index == len(word):
                if not node.end: return False
                else: present, node.end = True, False
                if not node.children:
                    return True
                return False
            c = word[index]
            if c not in node.children: return
            if deletor(node.children[c],index+1):
                del node.children[c]
                if not node.children: return True
                return False
            return False
        deletor(self.root,0)
        return present
    
    def deletepath(self, prefix: str, confirm=100) -> int:
        """Removes all words starting with the prefix from the Trie. Use "" to clear the entire Trie"""
        words = self.path_wordlist(prefix)
        if confirm and len(words) >= confirm:
            try:
                x = input(f"Are you sure? You are about to delete {len(words)} words! Type yes to confirm, or anything else to stop. ")
                if x != "yes": return 0
            except (KeyboardInterrupt,EOFError):
                print("\n Deletion cancelled.")
                return 0
        total = len(words)
        for word in words:
            self.delete(word)
        return total

    def path_wordlist(self, prefix: str) -> list:
        """Returns all the words starting with the prefix."""
        def collect_words(node,prefix,words):
            if node.end: words.append(prefix)
            for c in sorted(node.children):
                collect_words(node.children[c], prefix+c, words)
        path = ""
        curr = self.root
        for c in prefix:
            if c not in curr.children: return []
            curr = curr.children[c]
            path += c
        words = []
        collect_words(curr, path, words)
        return words
    
    def size(self) -> int:
        """Returns the amount of words in the Trie. Same as the size attribute"""
        n = 0
        def counter(node):
            nonlocal n
            if node.end: n += 1
            for c in node.children:
                counter(node.children[c])
        counter(self.root)
        return n
    
    def mostcommon(self, n=5, info=False) -> list:
        """Returns the n most common letters in the Trie."""
        common = {}
        q = deque([self.root])
        while q:
            node = q.popleft()
            for c in node.children:
                common[c] = common.get(c,0) + 1
                q.append(node.children[c])
        common_sorted = sorted(common.items(), key=lambda x: x[1], reverse=True)
        n = min(n, len(common_sorted))
        if not info:
            return [common_sorted[i][0] for i in range(n)]
        else:
            total = sum(count for _, count in common_sorted)
            result = []
            for letter,count in common_sorted[:n]:
                percentage = (count*10000 / total) // 1 / 100
                result.append((letter,percentage))
            return result

    def uniqueword(self, length: int = 5, banned: str = None, number: int = 1) -> list:
        """Returns a list of n words with unique letters of specified length in alphabetical order."""
        all_words = self.normal_wordlist()
        res = []
        for word in all_words:
            if len(word) == length and len(set(word)) == length:
                if banned and any(c in banned for c in word): continue
                res.append(word)
            if len(res) >= number:
                break
        return res
    
    def uniqueword_common(self, length: int = 5, banned: str = None, number: int = 1) -> list:
        """Returns a list of n words with unique letters of specified length based on most common English letters."""
        common = self.mostcommon(26)
        if banned:
            common = [common[i] for i in range(len(common)) if common[i] not in banned]
        if len(common) < length:
            return []
        all_words = self.normal_wordlist()
        res = []
        for word in all_words:
            if len(word) == length and len(set(word)) == length:
                if banned and any(c in banned for c in word): continue
                score = sum(1 for c in word if c in common[:length])
                res.append((score,word))
        res = sorted(res, key=lambda x: x[0], reverse=True)[:number]
        return [word for _, word in res]
    
    def spellcheck(self, check, info=False) -> str:
        """Returns the closest word in the Trie based on ordinal difference."""
        all_words = self.normal_wordlist()
        def distance(word, word2):
            if len(word) == len(word2):
                return sum(abs(ord(word[i])-ord(word2[i])) for i in range(len(word)))
            else:
                length = min(len(word),len(word2))
                distance = sum(abs(ord(word[i])-ord(word2[i])) for i in range(length))
                distance += sum(ord(char) for char in word[length:]) + sum(ord(char) for char in word2[length:])
                return distance
        best = (float("inf"), None)
        for length_diff in range(len(check)+1):
            candidates = [w for w in all_words if abs(len(w) - len(check)) == length_diff]
            for candidate in candidates:
                dist = distance(check, candidate)
                if dist < best[0]: best = (dist, candidate)
            if best[1]: break
        if not best[1]: return False
        if info: return best
        return best[1]

    def spellcheck2(self, check, options=1) -> str:
        """Returns the closest word in the Trie based on the Ratcliff/Obershelp algorithm."""
        all_words = self.normal_wordlist()
        matches = gcm(check, all_words, n=options,cutoff=0.25)
        if not matches: return False
        if options == 1: return matches[0]
        return matches
    
    def wordle(self, word, present=None, banned=None):
        all_words = self.wordlist(word)
        options = []
        for w in all_words:
            if banned and any(l in set(w) for l in set(banned)): continue
            if present and any(l not in set(w) for l in set(present)): continue
            options.append(w)
        return options




