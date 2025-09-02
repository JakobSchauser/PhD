#!/usr/bin/env python3
"""
Simple test to verify the conda environment is working
"""

print("🚀 Testing Conda Environment Setup")
print("=" * 50)

try:
    import numpy as np
    print("✅ NumPy imported successfully:", np.__version__)
except ImportError:
    print("❌ NumPy import failed")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError:
    print("❌ Matplotlib import failed")

try:
    import pandas as pd
    print("✅ Pandas imported successfully:", pd.__version__)
except ImportError:
    print("❌ Pandas import failed")

try:
    import seaborn as sns
    print("✅ Seaborn imported successfully")
except ImportError:
    print("❌ Seaborn import failed")

# Test if we can create a simple plot
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, label='sin(x)')
    ax.set_title('Test Plot - Environment Working!')
    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('environment_test.png', dpi=150, bbox_inches='tight')
    print("✅ Test plot created: environment_test.png")
    plt.close()
    
except Exception as e:
    print(f"❌ Plot creation failed: {e}")

print("\n🎯 Environment test complete!")
print("If all imports succeeded, the environment is ready for Hugging Face exercises.")
