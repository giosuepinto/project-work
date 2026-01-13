import sys
import os
import time
import json
import logging
import numpy as np

# --- CONFIGURAZIONE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

try:
    from Problem import Problem
    from s342711 import solution  # Assicurati che il nome sia corretto
    from plot_utils import plot_massive_results
except ImportError as e:
    print(f"Error Import: {e}")
    sys.exit(1)

# --- CONFIGURAZIONE SCENARI ---
N_VALUES = [10, 20, 50, 100, 200, 500, 1000]

SCENARIOS_TEMPLATE = [
    # 1. TSP-LIKE (Solo Distanza)
    {"name": "TSP-like", "alpha": 0.0, "beta": 1.0, "density": 0.2},
    {"name": "TSP-like", "alpha": 0.0, "beta": 1.0, "density": 0.5},
    {"name": "TSP-like", "alpha": 0.0, "beta": 1.0, "density": 1.0},
    
    # 2. LINEAR (Peso conta linearmente)
    {"name": "Linear", "alpha": 1.0, "beta": 1.0, "density": 0.2},
    {"name": "Linear", "alpha": 1.0, "beta": 1.0, "density": 0.5},
    {"name": "Linear", "alpha": 1.0, "beta": 1.0, "density": 1.0},
    
    # 3. LINEAR HEAVY (Peso conta doppio)
    {"name": "Linear2", "alpha": 2.0, "beta": 1.0, "density": 0.2},
    {"name": "Linear2", "alpha": 2.0, "beta": 1.0, "density": 0.5},
    {"name": "Linear2", "alpha": 2.0, "beta": 1.0, "density": 1.0},
    
    # 4. CONCAVE (Economie di scala)
    {"name": "Concave", "alpha": 1.0, "beta": 0.5, "density": 0.2},
    {"name": "Concave", "alpha": 1.0, "beta": 0.5, "density": 0.5},
    {"name": "Concave", "alpha": 1.0, "beta": 0.5, "density": 1.0},
    
    # 5. CONCAVE EXTREME
    {"name": "Concave2", "alpha": 2.0, "beta": 0.5, "density": 0.2},
    {"name": "Concave2", "alpha": 2.0, "beta": 0.5, "density": 0.5},
    {"name": "Concave2", "alpha": 2.0, "beta": 0.5, "density": 1.0},
    
    # 6. CONVEX (Penalità esponenziale)
    {"name": "Convex", "alpha": 1.0, "beta": 2.0, "density": 0.2},
    {"name": "Convex", "alpha": 1.0, "beta": 2.0, "density": 0.5},
    {"name": "Convex", "alpha": 1.0, "beta": 2.0, "density": 1.0},
    
    # 7. CONVEX EXTREME
    {"name": "Convex2", "alpha": 1.5, "beta": 2.5, "density": 0.2},
    {"name": "Convex2", "alpha": 1.5, "beta": 2.5, "density": 0.5},
    {"name": "Convex2", "alpha": 1.5, "beta": 2.5, "density": 1.0},
]

OUTPUT_DIR = os.path.join(current_dir, 'results', 'massive')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER DI VALIDAZIONE (Uguali a benchmark_single) ---
def validate_solution(problem, solution_path):
    errors = []
    if not solution_path or len(solution_path) < 2: return False, ["Empty path"]
    if solution_path[0] != (0, 0): errors.append("Start != (0,0)")
    if solution_path[-1] != (0, 0): errors.append("End != (0,0)")
        
    collected = {n: 0.0 for n in problem.graph.nodes}
    avail = {n: problem.graph.nodes[n]['gold'] for n in problem.graph.nodes}
    prev = 0
    
    for idx, (curr, gold) in enumerate(solution_path[1:], 1):
        if curr != prev and not problem.graph.has_edge(prev, curr):
            errors.append(f"Missing edge {prev}->{curr}")
        if gold < 0: errors.append(f"Negative gold at {curr}")
        collected[curr] += gold
        if collected[curr] > avail[curr] + 1e-5:
            errors.append(f"Over-collected at {curr}")
        prev = curr
            
    if errors: return False, errors
    return True, []

def calculate_official_cost(problem, solution_path):
    total = 0.0
    load = 0.0
    curr = 0
    for next_node, gold in solution_path[1:]:
        if next_node != curr:
            total += problem.cost([curr, next_node], load)
        load += gold
        if next_node == 0: load = 0.0
        curr = next_node
    return total

# --- MAIN LOOP MASSIVO ---
def run_massive_benchmark():
    print(f"STARTING MASSIVE BENCHMARK SUITE")
    print(f"Scenarios per N: {len(SCENARIOS_TEMPLATE)}")
    print(f"N Values: {N_VALUES}")
    
    for n in N_VALUES:
        print(f"\n{'='*60}")
        print(f">>> PROCESSING N = {n} <<<")
        print(f"{'='*60}")
        
        results_for_n = {}
        
        for sc in SCENARIOS_TEMPLATE:
            unique_name = f"{sc['name']} (d={sc['density']})"
            print(f"{'-'*60}")
            print(f"{unique_name:<20} | Alpha={sc['alpha']} Beta={sc['beta']} Density={sc['density']}\n", flush=True)
            
            try:
                # 1. Istanzia
                p = Problem(n, alpha=sc['alpha'], beta=sc['beta'], density=sc['density'], seed=42)
                
                # 2. Baseline
                t0 = time.time()
                base_cost = p.baseline()
                dur_base = time.time() - t0
                print(f"Baseline Cost: {base_cost:.2f} (Time: {dur_base:.2f}s)")
                # 3. Solver
                t0 = time.time()
                sol_path = solution(p)
                dur_solver = time.time() - t0
                print(f"My Solver Time: {dur_solver:.2f}s")
                
                # 4. Validazione & Costo
                print(f"Validating Solution...", end=" ", flush=True)
                t0 = time.time()
                is_valid, errs = validate_solution(p, sol_path)
                dur_val = time.time() - t0
                print(f"Done ({dur_val:.2f}s).")
                
                if not is_valid:
                    print(f"INVALID ({len(errs)} errs)")
                    solver_cost = float('inf')
                else:
                    solver_cost = calculate_official_cost(p, sol_path)
                    multiplier = base_cost / solver_cost if solver_cost > 0 else 1.0
                    print(f"VALID -> | Baseline: {base_cost:.2f}| MySolver: {solver_cost:.2f}| Time: {dur_solver:.2f}s | Final Increment: {multiplier:.2f}x\n")
                
                # 5. Store Data
                results_for_n[unique_name] = {
                    "group": sc['name'],
                    "density": sc['density'],
                    "alpha": sc['alpha'],
                    "beta": sc['beta'],
                    "baseline": base_cost,
                    "solution": solver_cost,
                    "time": dur_solver,
                    "valid": is_valid,
                    "improvement_x": base_cost / solver_cost if is_valid and solver_cost > 0 else 0.0
                }
                
            except Exception as e:
                print(f"CRASH: {e}")
        
        # Salvataggio JSON per questo N
        json_path = os.path.join(OUTPUT_DIR, f"results_N{n}.json")
        with open(json_path, 'w') as f:
            json.dump(results_for_n, f, indent=4)
            
        # Generazione Grafico
        print(f"Generating Plot for N={n}...", end=" ")
        plot_massive_results(results_for_n, n, os.path.join(OUTPUT_DIR, 'plots'))
        print("Done.")

if __name__ == "__main__":
    run_massive_benchmark()