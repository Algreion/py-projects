import random
import pygame
import time
import os
pygame.font.init()
pygame.mixer.init()

x = 0
def pure_bubblesort(t):
    n = 0
    swapped = True
    iterations = 1
    while swapped:
        swapped = False # Assumes we won't swap in this iteration
        for i in range(len(t)-iterations): # The range gets smaller every iteration
            n += 1
            if t[i] > t[i+1]: # If it's bigger than the one after, swap them
                n += 1
                swapped = True # This restarts the loop, if no swaps happen the loop ends
                t[i],t[i+1] = t[i+1],t[i]
        iterations += 1
    return n
def pure_insertion_sort(t):
    n = 0
    for i in range(1,len(t)):
        n += 1
        current = t[i] # Saves the state of the current element
        while True:
            n += 1
            if i == 0 or t[i-1] < current: # If the element before is lower or we are at t[0]
                break # Continue to the next element
            t[i] = t[i-1] # Else swap them
            i -= 1
            t[i] = current # Do so until ^
    return n
def pure_mergesort(t): # This is the recursive part, splits it into two
    global x
    x = x+1
    if len(t) == 1: # Base case, this works because [n] is already sorted
        return t
    mid = len(t)//2
    t1 = t[:mid] # Left
    t2 = t[mid:] # Right
    t1 = pure_mergesort(t1) 
    t2 = pure_mergesort(t2)
    return pure_merge(t1,t2)
def pure_merge(t1,t2):
    global x
    x += 1
    t = []
    while len(t1) > 0 and len(t2) > 0: # While both lists have elements
        x += 1
        if t1[0] > t2[0]:
            t.append(t2.pop(0)) # If the first is bigger, append the second
        else:
            t.append(t1.pop(0))
    while len(t1) > 0: # Picks up anything left in the other list, which is already sorted
        x += 1
        t.append(t1.pop(0))
    while len(t2) > 0:
        x += 1
        t.append(t2.pop(0))
    return t
def pure_selection_sort(t): # Made all by myself!
    res = []
    n = 0
    while len(t) != 0:
        n += 1
        s = float("inf")
        for i,j in enumerate(t):
            n += 1
            if j < s:
                s = j
                index = i
        res.append(t.pop(index))
    return n
def sort_tester(func,iterations=100,size=100,result=False):
    global x
    step_list=[]
    x = 0
    test = [3,2,1]
    func(test)
    if x > 0:
        recursive = True
    else:
        recursive = False
    for attempt in range(iterations):
        x = 0
        t = [n+1 for n in range(size)]
        random.shuffle(t)
        if recursive:
            t = func(t)
            steps = x
        else: 
            steps = func(t)
        step_list.append(steps)
    if not all(t[i] <= t[i+1] for i in range(len(t)-1)):
        print(f"{func.__name__} didn't work, the array is still unsorted! Take a look:")
        print(t)
        if result:
            return t
    average = sum(step_list)//iterations
    if not result:
        print(f"{func.__name__} took an average of {sum(step_list)//iterations} steps to sort an array of {size} elements.")
        print(f"The best case scenario took {min(step_list)} steps, the worst case took {max(step_list)}!")
        print(f"That is an efficiency of around {average/size} steps per element.")
    x = 0
    if result:
        return step_list
def compare(func1,func2,iterations=100,size=100):
    step_list1 = sort_tester(func1,iterations,size,result=True)
    step_list2 = sort_tester(func2,iterations,size,result=True)
    if sum(step_list1) == int((size)*(size+1)/2) or sum(step_list2) == (size)*(size+1)//2:
        return
    average1,average2 = sum(step_list1)//iterations,sum(step_list2)//iterations
    print(f"{func1.__name__} took an average of {average1} steps, {func2.__name__} took {average2}.")
    print(f"The best/worse cases for {func1.__name__} were {min(step_list1)}/{max(step_list1)}.")
    print(f"The best/worse cases for {func2.__name__} were {min(step_list2)}/{max(step_list2)}.")
    if average1 >= average2:
        percentage = average1/average2*100-100
        percentage = float("%.2f" % percentage)
        print(f"{func1.__name__} is +{percentage}% better than {func2.__name__} on average.")
        print(f"The best case for {func2.__name__} took {int(min(step_list2)/min(step_list1)*100)}% less steps than {func1.__name__}!")
        print(f"The worst case for {func2.__name__} took {int(max(step_list2)/max(step_list1)*100)}% less steps than {func1.__name__}!")
    else:
        percentage = average2/average1*100-100
        percentage = float("%.2f" % percentage)
        print(f"{func2.__name__} is +{percentage}% better than {func1.__name__} on average.")
        print(f"The best case for {func1.__name__} took {int(min(step_list1)/min(step_list2)*100)}% of {func2.__name__}!")
        print(f"The worst case for {func1.__name__} took {int(max(step_list1)/max(step_list2)*100)}% less steps than {func2.__name__}!")


# Actual cool stuff here:

WIDTH, HEIGHT = 1000, 600
TOTAL_RECTS = 100
RECT_WIDTH = 800//TOTAL_RECTS
RECT_SPACING = 2 if TOTAL_RECTS < 150 else 1
FONT = pygame.font.SysFont("verdana", 20)
RECT_COLOR = "cadetblue1"
SORTED_COLOR = "white"
TXT_COLOR = "white"
BG = "black"
TIME = 0.01
local = os.path.dirname(__file__)
BEEP = pygame.mixer.Sound(os.path.join(local, "blip.wav"))
WOP = pygame.mixer.Sound(os.path.join(local, "wop.wav"))

def generate_array():
    arr = list(range(1, TOTAL_RECTS + 1))
    random.shuffle(arr)
    return arr

def draw(win, arr, end=False, sorted_indices=set(), sort=None,c=0):
    pygame.event.pump()
    win.fill(BG)
    if end:
        sorted_indices = set(range(len(arr)))
    if sort:
        txt = FONT.render("Algorithm: " + sort, True, TXT_COLOR)
        win.blit(txt, (10, 10))
    win.blit(FONT.render("Total: " + str(TOTAL_RECTS), True, TXT_COLOR), (10, 35))
    total_width = TOTAL_RECTS * (RECT_WIDTH + RECT_SPACING)
    start_x = (WIDTH - total_width) // 2
    for i, val in enumerate(arr):
        height = int((val / TOTAL_RECTS) * (HEIGHT - 100))
        x = start_x + i * (RECT_WIDTH + RECT_SPACING)
        y = HEIGHT - height
        color = SORTED_COLOR if i in sorted_indices else RECT_COLOR
        pygame.draw.rect(win, color, (x, y, RECT_WIDTH, height))
        if end:
            time.sleep(TIME//2)
            pygame.display.update()
    if end: pygame.mixer.Sound.play(WOP)
    pygame.display.update()

def bubblesort(win, t, sound):
    sorted_indices = set()
    swapped = True
    iterations = 1
    while swapped:
        swapped = False # Assume we won't swap
        for i in range(len(t)-iterations):
            sorted_indices.add(i)
            if t[i] > t[i+1]:
                swapped = True
                t[i],t[i+1] = t[i+1],t[i]
                draw(win, t, False, sorted_indices, sort="Bubblesort")
                time.sleep(TIME)
            sorted_indices.remove(i)
        if sound: pygame.mixer.Sound.play(BEEP)
        sorted_indices.add(len(t)-iterations)
        iterations += 1
    draw(win, t, True, sorted_indices, sort="Bubblesort")
    return t
def insertionsort(win, t, sound):
    for i in range(1,len(t)):
        current = t[i]
        while True:
            if i == 0 or t[i-1] < current:
                break
            t[i] = t[i-1]
            i -= 1
            t[i] = current
            draw(win, t, False, set([i]), sort="Insertion Sort")
            time.sleep(TIME)
        if sound: pygame.mixer.Sound.play(BEEP)
    sorted_indices = set(range(len(t)))
    draw(win, t, True, sorted_indices, sort="Insertion Sort")
    return t
def quicksort(win, arr, sound, start=None, end=None):
    if start is None:
        start = 0
        end = len(arr) - 1
    
    def partition(start, end):
        pivot = arr[end]
        i = start - 1
        
        for j in range(start, end):
            if arr[j] <= pivot: # Put everything smaller on the left, bigger on the right
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                draw(win, arr, False, set([i, j]), sort="Quick Sort")
                time.sleep(TIME)
        arr[i + 1], arr[end] = arr[end], arr[i + 1] # Put the pivot in its correct position
        draw(win, arr, False, set([i + 1, end]), sort="Quick Sort")
        if sound: pygame.mixer.Sound.play(BEEP)
        time.sleep(TIME)
        return i + 1 # Return pivot's index

    if start < end:
        pivot = partition(start, end)
        quicksort(win, arr, sound, start, pivot - 1) # Sort everything on the left recursively
        quicksort(win, arr, sound, pivot + 1, end) # Same with right
    
    if start == 0 and end == len(arr) - 1: # First call stack, naturally finishes last
        draw(win, arr, True, set(range(len(arr))), sort="Quick Sort")
    return arr
def mergesort(win, arr, sound, start=None, end=None):
    if start is None:
        start = 0
        end = len(arr)
        
    if end - start > 1: # Base case is single element [n], already sorted
        mid = (start + end) // 2
        mergesort(win, arr, sound, start, mid)
        if sound: pygame.mixer.Sound.play(BEEP)
        mergesort(win, arr, sound, mid, end)
        if sound: pygame.mixer.Sound.play(BEEP)
        
        left = arr[start:mid]
        right = arr[mid:end]
        i = start
        j = 0
        k = 0
        
        while j < len(left) and k < len(right): # Put the smallest in the correct position in array
            if left[j] <= right[k]:
                arr[i] = left[j]
                j += 1
            else:
                arr[i] = right[k]
                k += 1
            draw(win, arr, False, set([i]), sort="Merge Sort")
            time.sleep(TIME)
            i += 1
            
        while j < len(left): # Cleanup remaining elements in left
            arr[i] = left[j]
            draw(win, arr, False, set([i]), sort="Merge Sort")
            time.sleep(TIME)
            i += 1
            j += 1
            
        while k < len(right): # And right
            arr[i] = right[k]
            draw(win, arr, False, set([i]), sort="Merge Sort")
            time.sleep(TIME)
            i += 1
            k += 1
    
    if start == 0 and end == len(arr):
        draw(win, arr, True, set(range(len(arr))), sort="Merge Sort")
    return arr
def selectionsort(win, arr, sound):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
            draw(win, arr, False, set([min_idx, j]), sort="Selection Sort")
            time.sleep(TIME)
        
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        if sound: pygame.mixer.Sound.play(BEEP)
        draw(win, arr, False, set([i, min_idx]), sort="Selection Sort")
        time.sleep(TIME)
    
    draw(win, arr, True, set(range(len(arr))), sort="Selection Sort")
    return arr
def shellsort(win, arr, sound):
    n = len(arr) 
    gap = n // 2 # Basically insertion sort
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i] # Stores the value of current element
            j = i
            while j >= gap and arr[j - gap] > temp: # Checks it against all others with distance gap
                arr[j] = arr[j - gap]
                draw(win, arr, False, set([j, j-gap]), sort="Shell Sort")
                time.sleep(TIME)
                j -= gap
            arr[j] = temp
            if sound and not i%5: pygame.mixer.Sound.play(BEEP)
            draw(win, arr, False, set([j]), sort="Shell Sort")
            time.sleep(TIME)
        gap //= 2 # Redo with half the gap
    
    draw(win, arr, True, set(range(len(arr))), sort="Shell Sort")
    return arr
def heapsort(win, arr, sound):
    def heapify(n, i): # Build an actual max heap
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            draw(win, arr, False, set([i, largest]), sort="Heap Sort")
            time.sleep(TIME)
            heapify(n, largest)

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1): # From first parent, heapify on the left
        heapify(n, i)
        if sound and not i%3: pygame.mixer.Sound.play(BEEP)

    for i in range(n - 1, 0, -1): # Position the max on the right, reapply max heap rule
        arr[0], arr[i] = arr[i], arr[0]
        draw(win, arr, False, set([0, i]), sort="Heap Sort")
        time.sleep(TIME)
        heapify(i, 0)
        if sound: pygame.mixer.Sound.play(BEEP)

    draw(win, arr, True, set(range(len(arr))), sort="Heap Sort")
    return arr
def cocktailsort(win, arr, sound):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    sorted = set()
    while swapped: # Bubblesort but slightly faster
        swapped = False

        # Forward pass (left to right)
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
                time.sleep(TIME)
            sorted.add(i+1)
            draw(win, arr, False, sorted, sort="Cocktail Sort")
            sorted.remove(i+1)
        if sound: pygame.mixer.Sound.play(BEEP)

        if not swapped: # We're done
            break

        swapped = False
        sorted.add(end)
        end -= 1 # Last element is sorted

        # Backward pass (right to left)
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
                time.sleep(TIME)
            sorted.add(i)
            draw(win, arr, False, sorted, sort="Cocktail Sort")
            sorted.remove(i)
        if sound: pygame.mixer.Sound.play(BEEP)

        sorted.add(start)
        start += 1 # First element is sorted

    draw(win, arr, True, set(range(len(arr))), sort="Cocktail Sort")
    return arr
def combsort(win, arr, sound):
    n = len(arr)
    gap = n
    shrink = 1.3
    sorted = False

    while not sorted: # Bubblesort but better, same idea as shellsort
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True

        for i in range(0, n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted = False
                draw(win, arr, False, set([i, i+gap]), sort="Comb Sort")
                if sound and not i%3: pygame.mixer.Sound.play(BEEP)
                time.sleep(TIME)
    
    draw(win, arr, True, set(range(len(arr))), sort="Comb Sort")
    return arr
def cyclesort(win, arr, sound):
    # O(n^2), O(1) memory | Unstable
    n = len(arr)
    
    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start

        for i in range(cycle_start + 1, n): # Find index for item
            if arr[i] < item:
                pos += 1
                time.sleep(TIME//1.5)
                draw(win, arr, False, set([cycle_start, pos]), sort="Cycle Sort")

        if pos == cycle_start: # Already in right position
            continue

        while item == arr[pos]: # Check for duplicates
            pos += 1

        arr[pos], item = item, arr[pos]
        if sound: pygame.mixer.Sound.play(BEEP)
        draw(win, arr, False, set([cycle_start, pos]), sort="Cycle Sort")
        
        time.sleep(TIME)

        while pos != cycle_start: # Loops the cycle
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
                    time.sleep(TIME//1.5)
                    draw(win, arr, False, set([cycle_start, pos]), sort="Cycle Sort")

            while item == arr[pos]:
                pos += 1

            arr[pos], item = item, arr[pos]
            if sound: pygame.mixer.Sound.play(BEEP)
            draw(win, arr, False, set([cycle_start, pos]), sort="Cycle Sort")
            time.sleep(TIME)
    
    draw(win, arr, True, set(range(len(arr))), sort="Cycle Sort")
    return arr
def bogosort(win, arr, sound):
    wait_time = random.uniform(1, 30)
    start_time = time.time()

    while time.time() - start_time < wait_time:
        random.shuffle(arr)
        draw(win, arr, False, sort="Bogosort")
        time.sleep(TIME)
    sorted_arr = sorted(arr)
    arr[:] = sorted_arr
    if sound: pygame.mixer.Sound.play(BEEP)
    draw(win, arr, False, set(range(len(arr))), sort="Bogosort")
    return arr
def pancakesort(win, arr, sound):
    def flip(arr, k):
        if sound: pygame.mixer.Sound.play(BEEP)
        left = 0
        while left < k:
            arr[left], arr[k] = arr[k], arr[left]
            draw(win, arr, False, set([left, k]), sort="Pancake Sort")
            time.sleep(TIME)
            left += 1
            k -= 1
    
    n = len(arr)
    for size in range(n-1, 0, -1):
        max_idx = 0
        for i in range(1, size+1):
            if arr[i] > arr[max_idx]:
                max_idx = i
        
        if max_idx != size:
            if max_idx != 0:
                flip(arr, max_idx)
                
            flip(arr, size)
    
    draw(win, arr, True, set(range(len(arr))), sort="Pancake Sort")
    return arr



def main(func=bubblesort):
    global BEEP, TIME
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sorting Visualizer")
    t = generate_array()
    sorter = func.__name__.capitalize()
    run = True
    sorting = False
    sound = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN and not sorting:
                if event.key == pygame.K_b:
                    func = bubblesort
                    sorter = "Bubblesort | O(n^2)"
                elif event.key == pygame.K_i:
                    func = insertionsort
                    sorter = "Insertion Sort | O(n^2)"
                elif event.key == pygame.K_q:
                    func = quicksort
                    sorter = "Quick Sort | O(n*log(n))"
                elif event.key == pygame.K_m:
                    func = mergesort
                    sorter = "Merge Sort | O(n*log(n))"
                elif event.key == pygame.K_s:
                    func = selectionsort
                    sorter = "Selection Sort | O(n^2)"
                elif event.key == pygame.K_h:
                    func = heapsort
                    sorter = "Heap Sort | O(n*log(n))"
                elif event.key == pygame.K_l:
                    func = shellsort
                    sorter = "Shell Sort | O(n*2log(n))"
                elif event.key == pygame.K_c:
                    func = cocktailsort
                    sorter = "Cocktail Sort | O(n^2)"
                elif event.key == pygame.K_o:
                    func = combsort
                    sorter = "Comb Sort | O(n^2)"
                elif event.key == pygame.K_y:
                    func = cyclesort
                    sorter = "Cycle Sort | O(n^2)"
                elif event.key == pygame.K_j:
                    func = bogosort
                    sorter = "Bogosort | O(n*n!)"
                elif event.key == pygame.K_p:
                    func = pancakesort
                    sorter = "Pancake Sort | O(n^2)"
                elif event.key == pygame.K_r:
                    t = generate_array()
                elif event.key == pygame.K_x:
                    sound = False
                elif event.key == pygame.K_0:
                    TIME = 0
                elif event.key == pygame.K_SPACE:
                    sorting = True
                    func(win, t, sound)
        if not sorting:
            draw(win, t, sort=sorter)
    pygame.quit()

if __name__ == "__main__":
    print("Welcome to the best sorting visualizer in town! Here are the commands: ")
    print("""\nSPACE: Start sorting | R: Randomize Array | X: Mute sounds | 0: Fastest sorting\nB: bubble_sort | I: insertion_sort | Q: quick_sort | M: merge_sort | S: selection_sort | H: heap_sort
L: shell_sort | C: cocktail_sort | O: comb_sort | Y: cycle_sort | J: bogosort | P: pancake_sort""")
    main()

