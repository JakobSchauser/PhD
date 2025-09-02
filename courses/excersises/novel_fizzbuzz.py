#!/usr/bin/env python3
"""
Novel FizzBuzz Implementation
An elegant generator-based approach using dictionary mapping.
"""

def fizzbuzz(n=100):
    """
    Generator-based FizzBuzz using dictionary mapping and modulo magic.
    
    This implementation is:
    - Memory efficient (generator)
    - Easily extensible (just add rules to the dictionary)
    - Clean and readable
    - Pythonic
    
    Args:
        n (int): Maximum number to generate FizzBuzz for
        
    Yields:
        str: FizzBuzz result for each number
    """
    rules = {3: "Fizz", 5: "Buzz"}
    
    for i in range(1, n + 1):
        result = "".join(word for divisor, word in rules.items() if i % divisor == 0)
        yield result or str(i)


def main():
    """Demonstrate the FizzBuzz implementation."""
    print("🎯 Elegant FizzBuzz Implementation\n")
    
    # Print first 100 numbers
    for i, result in enumerate(fizzbuzz(100), 1):
        print(f"{i:3}: {result}")


if __name__ == "__main__":
    main()
