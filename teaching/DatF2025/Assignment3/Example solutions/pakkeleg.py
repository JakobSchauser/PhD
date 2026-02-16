import random



N_good = 26

class Packages(list[bool]):
    """
    A simple subclass of list to represent packages.
    Each package is represented as a boolean: True for good, False for bad.
    """
    pass



def sister_strategy(packages):
    s1 = 0

    for i in range(100):
        if packages[i]:
            s1 += 1
        if s1 == 2:
            return i
        
    return "asdasd"

def nonbinary_strategy(packages):
    s2 = 0

    for i in range(50):
        if packages[2*i]:
            s2 += 1
        if s2 == 2:
            return i
        
    for ii in range(50):
        if packages[2*i+1]:
            s2 += 1
        if s2 == 2:
            return ii+i 
    return "asdasd"

# def sister_strategy(packages : Packages) -> int:
#     """
#     Sister's strategy: Open every package from first to last.
#     """
#     minutes = 0
#     good_found = 0
#     for i in range(len(packages)):
#         minutes += 1
#         if packages[i]:
#             good_found += 1

#         if good_found == N_good:
#             return minutes
#     return minutes  # Should not reach if at least 2 good

def brother_strategy(packages : Packages) -> int:
    """
    Brother's strategy: Open a random unopened package each time.
    """
    unopened = set(range(len(packages)))
    minutes = 0
    good_found = 0
    while good_found < N_good and unopened:
        idx = random.choice(list(unopened))
        unopened.remove(idx)
        minutes += 1
        if packages[idx]:
            good_found += 1
    return minutes

def nonbinary_strategy(packages : Packages) -> int:
    """
    Non-binary sibling's strategy: Open all odd packages first, then even ones.
    """
    minutes = 0
    good_found = 0
    # Odd indices first

    # Even indices
    for i in range(0, len(packages), 2):
        minutes += 1
        if packages[i]:
            good_found += 1
            if good_found == N_good:
                return minutes
            
    for i in range(1, len(packages), 2):
        minutes += 1
        if packages[i]:
            good_found += 1
            if good_found == N_good:
                return minutes
    return minutes

# Optional: Your own strategy


def nonbinary_strategy2(packages):
    s2 = 0

    for i in range(50):
        if packages[2*i]:
            s2 += 1
        if s2 == 2:
            return i
        
    for ii in range(50):
        if packages[2*i+1]:
            s2 += 1
        if s2 == 2:
            return ii+i 
    return "asdasd"