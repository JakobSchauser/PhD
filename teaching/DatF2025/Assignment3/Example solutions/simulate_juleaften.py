from pakkeleg import sister_strategy, brother_strategy, nonbinary_strategy, Packages, my_strategy
import random

# Parameters
N_PACKAGES = 100
N_GOOD = 26
N_TRIALS = 10000

# Strategies
strategies = {
    'Sister': sister_strategy,
    # 'Brother': brother_strategy,
    'Non-binary': nonbinary_strategy,
    # 'my_strategy': my_strategy
    
    
}

# Wins counter
wins = {name: 0 for name in strategies}

def make_packages(n_packages: int, n_good: int) -> Packages:
    """
    Create a Packages object with n_packages total packages,
    of which n_good are good (True) and the rest are bad (False).
    The good packages are placed randomly.
    """
    packages = Packages([False] * n_packages)
    good_indices = random.sample(range(n_packages), n_good)
    for idx in good_indices:
        packages[idx] = True
    return packages

for trial in range(N_TRIALS):
    # Generate packages: Packages with N_GOOD good ones randomly placed
    packages = make_packages(N_PACKAGES, N_GOOD)
    
    # Run each strategy
    times = {}
    for name, strategy in strategies.items():
        times[name] = strategy(packages)
    
    
    # Find the minimum time
    min_time = min(times.values())
    
    for name in strategies:
        if times[name] == min_time:
            wins[name] += 1

# Print results
print("Simulation Results:")
print(f"Total trials: {N_TRIALS}")
print(f"Packages: {N_PACKAGES}, Good packages: {N_GOOD}")
print()
for name in strategies:
    percentage = (wins[name] / N_TRIALS) * 100
    print(f"{name}: {wins[name]} wins ({percentage:.2f}%)")

# Determine who is most likely
max_wins = max(wins.values())
best = [name for name, w in wins.items() if w == max_wins]
if len(best) == 1:
    print(f"\nThe strategy most likely to win is: {best[0]}")
else:
    print(f"\nThe strategies most likely to win are: {', '.join(best)}")

# Does strategy matter?
total_unique_wins = sum(1 for w in wins.values() if w > 0)
if total_unique_wins > 1:
    print("Strategies do matter; some are better than others.")
else:
    print("Strategies are evenly likely.")