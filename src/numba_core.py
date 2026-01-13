import numpy as np
from numba import jit

@jit(nopython=True)
def prins_split_algorithm(tour, dist_matrix, penalty_matrix, golds, alpha, beta):
    """
    Algoritmo di Split (Prins) Corretto.
    Calcola il costo incrementale arco per arco, allineandosi a Problem.py.
    """
    n = len(tour)
    MAX_VAL = 1e30
    V = np.full(n + 1, MAX_VAL)
    V[0] = 0.0
    P = np.zeros(n + 1, dtype=np.int32)
    
    # Parametri
    look_ahead = 150
    
    for i in range(n):
        if V[i] >= MAX_VAL: continue
        
        current_load = 0.0
        segment_cost = 0.0
        curr_node = 0 # Partenza dal deposito (virtual node 0)
        
        limit = min(n, i + look_ahead)
        
        for j in range(i, limit):
            next_node = tour[j]
            
            # 1. Costo Arco (curr -> next) con il carico CORRENTE
            d = dist_matrix[curr_node, next_node]
            p = penalty_matrix[curr_node, next_node]
            
            # Formula Problem.py: dist + (alpha * load * dist_weight)**beta
            # Nota: Problem.py usa (alpha * dist * weight)**beta
            # Se beta=1 (Linear): dist + alpha * dist * load
            # Se beta!=1: dist + (alpha * load)**beta * p
            # Assumiamo p pre-calcolato come w**beta
            
            step_cost = d
            if alpha > 0 and current_load > 0:
                step_cost += ((alpha * current_load) ** beta) * p
            
            segment_cost += step_cost
            
            # 2. Aggiorna stato per il prossimo passo
            current_load += golds[next_node]
            curr_node = next_node
            
            # 3. Costo Chiusura (next -> 0) con il carico FINALE accumulato
            d_ret = dist_matrix[curr_node, 0]
            p_ret = penalty_matrix[curr_node, 0]
            
            ret_cost = d_ret
            if alpha > 0 and current_load > 0:
                ret_cost += ((alpha * current_load) ** beta) * p_ret
            
            total_trip_cost = segment_cost + ret_cost
                
            # Bellman equation
            if V[i] + total_trip_cost < V[j+1]:
                V[j+1] = V[i] + total_trip_cost
                P[j+1] = i
                
    return V[n], P

@jit(nopython=True)
def fast_2opt_stochastic(tour, dist_matrix, penalty_matrix, golds, alpha, beta, max_attempts=1000):
    """
    2-Opt Stocastico aggiornato per usare il nuovo Prins.
    """
    best_tour = tour.copy()
    best_cost, _ = prins_split_algorithm(best_tour, dist_matrix, penalty_matrix, golds, alpha, beta)
    
    n = len(tour)
    if n < 3: return best_tour, best_cost # Nulla da ottimizzare
    
    candidate_tour = np.empty_like(best_tour)
    
    # Adatta i tentativi alla dimensione
    attempts = max_attempts
    
    for _ in range(attempts):
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        
        # Inversione segmenti (copia veloce)
        candidate_tour[:] = best_tour[:]
        # Inverti slice i:j
        # Numba supporta array slicing efficiente
        head = i
        tail = j
        while head < tail:
            candidate_tour[head], candidate_tour[tail] = candidate_tour[tail], candidate_tour[head]
            head += 1
            tail -= 1
            
        new_cost, _ = prins_split_algorithm(candidate_tour, dist_matrix, penalty_matrix, golds, alpha, beta)
        
        if new_cost < best_cost - 1e-6:
            best_cost = new_cost
            best_tour[:] = candidate_tour[:]
            
    return best_tour, best_cost

@jit(nopython=True)
def ordered_crossover_numba(parent1, parent2):
    size = len(parent1)
    child = np.full(size, -1, dtype=np.int32)
    cut1 = np.random.randint(0, size)
    cut2 = np.random.randint(0, size)
    start, end = min(cut1, cut2), max(cut1, cut2)
    child[start:end] = parent1[start:end]
    current_pos = end
    for gene in parent2:
        is_in = False
        for k in range(start, end):
            if child[k] == gene:
                is_in = True
                break
        if not is_in:
            if current_pos >= size: current_pos = 0
            child[current_pos] = gene
            current_pos += 1
    return child

@jit(nopython=True)
def swap_mutation_numba(tour, rate):
    if np.random.random() > rate: return tour
    n = len(tour)
    idx1, idx2 = np.random.randint(0, n), np.random.randint(0, n)
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour