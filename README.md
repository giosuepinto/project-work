
# CREDITS
This project work was created and developed by **Giosuè Pinto s342711** and **Francesco Palmisani s343429**


## Project Structure

```text
PROJECT_ROOT/
│
├── s342711.py              # MAIN SOLUTION FILE (Use this for grading)
├── Problem.py              # Problem definition (Provided by Professor)
├── base_requirements.txt   # Python dependencies
├── README.md               # This file
├── REPORT.md               # Technical Report about the implementation
│
├── src/                    # Optimization Engine (The "Brain")
│   ├── __init__.py         # Package marker
│   ├── numba_core.py       # JIT-compiled algorithms (Prins, 2-Opt)
│   └── problem_utils.py    # Floyd-Warshall & Graph Augmentation
│
└── test/                   # Benchmark & Validation Suite
    ├── benchmark_single.py # Runs a single scenario with strict validation
    ├── benchmark_massive.py# Runs the full test suite (N=10 to N=1000)
    ├── plot_utils.py       # Plotting utilities
    └── results/            # Auto-generated JSON reports and Charts
```

---

## Installation

Ensure you have Python 3.8+ installed (Up to python 3.13 for Numba!). Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

*Note: `icecream` is required because it is imported inside `Problem.py` for debugging purposes.*

---

## How to Run

### 1. Standard Execution (Professor's Mode)
To use the solver as intended in the examination pipeline, simply import the `solution` function from `s342711.py` and pass a `Problem` instance.

```python
from Problem import Problem
from s342711 import solution

# Generate a random problem instance
p = Problem(num_cities=50)

# Run the solver
path = solution(p)

print(f"Solution found with {len(path)} steps.")
```

### 2. Single Scenario Benchmark (Development Mode)
To run a specific test case (e.g., N=50, Convex), verify its validity (topological consistency, gold collected), and compare it against the Official Baseline:

```bash
python test/benchmark_single.py
```
1. Tip: You can modify the `SCENARIO` dictionary inside `test/benchmark_single.py` to test different parameters (Alpha, Beta, Density).*
2. Tip: The file `s342711.py` contains the function `solution` that acts as a wrapper for the solve method. In the solve method it's possible to set the `time_limit` representing the time limitation for the algorithm to run. In the experiments it was set to `600 seconds`, in most of the cases the algorithm didn't exeed the time limit and early stopped.

### 3. Massive Benchmark Suite (Full Report)
To reproduce the experimental results and generate performance charts for all 21 scenarios across sizes $N \in \{10, \dots, 1000\}$:

```bash
python test/benchmark_massive.py
```
*Artifacts (JSON data and PNG charts) will be saved in the `test/results/massive/` folder.*

---

## Important Note on Benchmark Execution Time

When running the scripts in the `test/` folder, you might notice the total execution time is significantly longer than the reported "Solver Time".

**This is expected.** The benchmark script performs three tasks sequentially:

1.  **Baseline Calculation :** It runs `problem.baseline()`. This method computes a "Hub & Spoke" Dijkstra path for *every single node* individually.
2.  **The Solver (FAST):** This is the execution of my solution (`s342711.py`). It typically takes sub-second times for small N and up to 600s for N=1000.
3.  **Strict Validation (SLOW):** The script iterates through the final path step-by-step to verify topological consistency (checking edge existence in NetworkX) and Gold accounting to ensure the solution is legal.

**Conclusion:** The wall-clock time of the benchmark script is dominated by the Verification tools for bigger N, NOT by the solver itself.

---

## Performance Summary

The solution implements a **Hybrid Architecture** designed for high-throughput evaluation:
* **Preprocessing:** Augmented Floyd-Warshall ($O(N^3)$ executed once).
* **Search:** Generational Genetic Algorithm with Elitism and specialized Seeding.
* **Evaluation:** Prins' Split Algorithm with Lookahead (Linear Complexity).
* **Local Search:** Stochastic 2-Opt (Numba Accelerated).

**Experimental Results:**
* **TSP-like Scenarios ($\alpha \approx 0$):** >20x improvement over baseline.
* **Convex Scenarios ($\beta > 1$):** >2.5x improvement (validating the Split Delivery logic).
* **Linear Scenarios:** 1.0x Tie (Robustness guarantee via Elitism).

**You can find more details about the implementation in the REPORT.md file**