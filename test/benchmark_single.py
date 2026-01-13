import sys
import os
import time
import json
import networkx as nx

# --- CONFIGURAZIONE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

try:
    from Problem import Problem
    from s342711 import solution  # Assicurati che il nome file sia corretto
    from plot_utils import plot_single_comparison
except ImportError as e:
    print(f"Error Import: {e}")
    sys.exit(1)

# --- CONFIGURA IL TUO SCENARIO ---
SCENARIO = {
    "N": 50,
    "DENSITY": 0.8,
    "ALPHA": 15.0,
    "BETA": 5.5,       
    "SEED": 42
}

OUTPUT_FOLDER = os.path.join(current_dir, 'results', 'single')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. VALIDATORE RIGOROSO (Simulazione Prof) ---
def validate_solution(problem, solution_path):
    """
    Verifica che la soluzione rispetti le regole del gioco:
    1. Inizia e finisce a (0,0).
    2. Adiacenza: Ogni passo (u, v) deve esistere nel grafo.
    3. Oro: Non puoi prendere oro che non c'è.
    4. Completezza: (Opzionale, ma controlliamo se hai preso tutto).
    """
    errors = []
    
    # Check 1: Struttura
    if not solution_path or len(solution_path) < 2:
        return False, ["Path too short or empty."]
    
    if solution_path[0] != (0, 0):
        errors.append(f"Start point invalid: {solution_path[0]} != (0,0)")
    if solution_path[-1] != (0, 0):
        errors.append(f"End point invalid: {solution_path[-1]} != (0,0)")
        
    # Check 2: Topologia & Oro
    total_gold_collected = 0
    available_gold = {n: problem.graph.nodes[n]['gold'] for n in problem.graph.nodes}
    collected_per_node = {n: 0.0 for n in problem.graph.nodes}
    
    prev_node = 0
    
    # Iteriamo dal secondo elemento in poi
    for idx, (curr_node, gold_taken) in enumerate(solution_path[1:], 1):
        
        # A. Verifica Adiacenza Fisica
        if curr_node != prev_node: # Se ci muoviamo
            if not problem.graph.has_edge(prev_node, curr_node):
                errors.append(f"Step {idx}: Non-existant Edge {prev_node} -> {curr_node}")
        
        # B. Verifica Oro
        if gold_taken < 0:
             errors.append(f"Step {idx}: Negative gold taken ({gold_taken})")
        
        collected_per_node[curr_node] += gold_taken
        
        if collected_per_node[curr_node] > available_gold[curr_node] + 1e-5: # tolleranza float
             errors.append(f"Step {idx}: Taken more gold than available at node {curr_node}")

        prev_node = curr_node

    # Check 3: Abbiamo preso tutto? (Opzionale ma utile)
    total_map_gold = sum(available_gold.values())
    total_user_gold = sum(collected_per_node.values())
    
    if abs(total_map_gold - total_user_gold) > 1.0:
        errors.append(f"Missing gold! Map: {total_map_gold:.2f}, Yours: {total_user_gold:.2f}")

    if errors:
        return False, errors
    return True, []

# --- 2. SCORING (Calcolo Costo) ---
def calculate_official_cost(problem, solution_path):
    """
    Calcola il costo usando la funzione problem.cost([u,v], weight).
    Assume che la soluzione sia già stata validata (topologia ok).
    """
    total_cost = 0.0
    current_load = 0.0
    current_node = 0
    
    # Skip start (0,0)
    for next_node, gold_taken in solution_path[1:]:
        
        # Calcolo costo spostamento (se c'è spostamento)
        if next_node != current_node:
            # Qui usiamo problem.cost che usa nx.path_weight
            # Dato che il validatore ha garantito l'arco, questo non deve crashare
            cost_step = problem.cost([current_node, next_node], current_load)
            total_cost += cost_step
        
        # Aggiorna carico
        current_load += gold_taken
        
        # Scarico alla base
        if next_node == 0:
            current_load = 0.0
            
        current_node = next_node
        
    return total_cost

# --- MAIN FLOW ---
def main():
    print(f"\n--- SIMULATION BENCHMARK ---")
    print(f"Scenario: N={SCENARIO['N']}, Density={SCENARIO['DENSITY']}, Alpha={SCENARIO['ALPHA']}, Beta={SCENARIO['BETA']}")
    
    # 1. ISTANZIA PROBLEMA
    p = Problem(
        SCENARIO['N'], 
        density=SCENARIO['DENSITY'], 
        alpha=SCENARIO['ALPHA'], 
        beta=SCENARIO['BETA'], 
        seed=SCENARIO['SEED']
    )
    
    # 2. RUN BASELINE (PROF)
    print("Running Official Baseline...", end=" ", flush=True)
    t0 = time.time()
    base_cost = p.baseline()
    t_base = time.time() - t0
    print(f"Done ({t_base:.2f}s). Cost: {base_cost:,.2f}")

    # 3. RUN SOLUZIONE
    print("Running My Solution...\n", flush=True)
    t0 = time.time()
    user_path = solution(p)
    t_sol = time.time() - t0
    print(f"Done ({t_sol:.2f}s).")
    
    # 4. VALIDAZIONE (Il momento della verità)
    print("Validating Solution...", end=" ", flush=True)
    t0 = time.time()
    is_valid, errors = validate_solution(p, user_path)
    t_sol = time.time() - t0
    print(f"Done ({t_sol:.2f}s).")
    
    if not is_valid:
        print("\nSOLUTION REJECTED!")
        for e in errors[:5]: print(f"   - {e}")
        if len(errors) > 5: print("   ... (other errors omitted)")
        user_cost = float('inf')
    else:
        print("VALID.")
        # 5. CALCOLO COSTO
        user_cost = calculate_official_cost(p, user_path)
        print(f"My Solution Cost: {user_cost:,.2f}")

    # 6. CONFRONTO E OUTPUT
    multiplier = base_cost / user_cost if user_cost > 0 else 0.0
    print(f"\nPERFORMANCE MULTIPLIER: {multiplier:.2f}x")
    
    # Grafico
    scenario_str = f"N={SCENARIO['N']} A={SCENARIO['ALPHA']} B={SCENARIO['BETA']}"
    plot_path = os.path.join(OUTPUT_FOLDER, f"chart_{scenario_str.replace(' ', '_')}.png")
    
    if is_valid:
        plot_single_comparison(base_cost, user_cost, scenario_str, plot_path)
    
    # Json
    json_path = plot_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            "scenario": SCENARIO,
            "valid": is_valid,
            "errors": errors,
            "baseline": base_cost,
            "solution": user_cost,
            "multiplier": multiplier,
            "time_solver": t_sol
        }, f, indent=4)
    print(f"Report saved to: {json_path}")

if __name__ == "__main__":
    main()