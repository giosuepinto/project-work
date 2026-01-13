import sys
import os
import time
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.problem_utils import process_graph_data
    from src.numba_core import (
        prins_split_algorithm, 
        fast_2opt_stochastic, 
        ordered_crossover_numba, 
        swap_mutation_numba
    )
except ImportError:
    from problem_utils import process_graph_data
    from numba_core import (
        prins_split_algorithm, 
        fast_2opt_stochastic, 
        ordered_crossover_numba, 
        swap_mutation_numba
    )

class Solution:
    def __init__(self, problem):
        self.problem = problem
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Pre-processing dei dati (Floyd-Warshall e Virtual Nodes)
        data = process_graph_data(problem.graph, self.alpha, self.beta)
        
        self.dist_matrix = data['dist_matrix']
        self.penalty_matrix = data['penalty_matrix']
        self.golds = data['golds']
        self.virtual_map = data['virtual_map']
        self.predecessors = data['predecessors']
        
        # Nodi su cui lavora il GA (escluso 0 che è deposito)
        self.nodes = np.arange(1, len(self.golds), dtype=np.int32)
        
    def _reconstruct_path(self, source, target):
        """Ricostruisce il percorso fisico usando i predecessori (Floyd-Warshall/Dijkstra)."""
        if source == target: return [source]
        path = []
        curr = target
        while curr != source:
            path.append(curr)
            prev = self.predecessors[source, curr]
            if prev == -9999: return [] # No path
            curr = prev
        path.append(source)
        return path[::-1]

    def solve(self, time_limit=20):
        start_time = time.time()
        
        # Caso banale: nessun cliente
        if len(self.nodes) == 0: return [(0, 0)]
        
        # --- CONFIGURAZIONE GA ---
        pop_size = 50
        elite_size = 4
        mutation_rate = 0.2
        # Parametri Memetici
        memetic_prob = 0.15      # Probabilità che un figlio subisca Local Search immediata
        memetic_attempts = 30    # Quanti swap provare nella Local Search interna
        
        # Parametri Early Stopping
        stagnation_limit = 300   # Generazioni senza miglioramenti prima di uscire
        stagnation_counter = 0
        
        population = []
        pop_fitness = [] # Lista di tuple (costo, split_array, tour)

        # --- 1. POPOLAZIONE INIZIALE (SEEDING) ---
        
        # A. Greedy Seed
        # Costruisce un tour basato sul Nearest Neighbor
        greedy_tour = []
        unvisited = set(self.nodes)
        curr = 0
        # Limite greedy per grafi enormi
        if len(self.nodes) < 3000:
            while unvisited:
                # Trova il più vicino (euristica semplice su dist_matrix)
                # Nota: Per velocità in Python puro usiamo min su un subset se necessario
                # Qui facciamo semplice scansione
                best_n = min(unvisited, key=lambda x: self.dist_matrix[curr, x])
                greedy_tour.append(best_n)
                unvisited.remove(best_n)
                curr = best_n
        else:
            greedy_tour = list(self.nodes) # Fallback random se troppo grande
            
        greedy_arr = np.array(greedy_tour, dtype=np.int32)
        c_greedy, s_greedy = prins_split_algorithm(greedy_arr, self.dist_matrix, self.penalty_matrix, self.golds, self.alpha, self.beta)
        population.append(greedy_arr)
        pop_fitness.append((c_greedy, s_greedy, greedy_arr))

        # B. Random Seeds
        while len(population) < pop_size:
            rand_tour = np.random.permutation(self.nodes).astype(np.int32)
            c, s = prins_split_algorithm(rand_tour, self.dist_matrix, self.penalty_matrix, self.golds, self.alpha, self.beta)
            population.append(rand_tour)
            pop_fitness.append((c, s, rand_tour))
            
        # Ordina per costo
        pop_fitness.sort(key=lambda x: x[0])
        
        best_global_cost = pop_fitness[0][0]
        best_global_split = pop_fitness[0][1]
        best_global_tour = pop_fitness[0][2]
        
        initial_best_cost = best_global_cost
        print(f"DEBUG: Initial Seed Cost: {best_global_cost:,.2f}")
        
        generation = 0
        
        # --- 2. CICLO EVOLUTIVO ---
        while time.time() - start_time < time_limit:
            generation += 1
            
            # Elitismo: Mantieni i migliori
            new_pop_fitness = pop_fitness[:elite_size]
            new_population = [x[2] for x in new_pop_fitness]
            
            # Generazione figli
            while len(new_population) < pop_size:
                # Selezione Torneo (veloce)
                # Scegliamo indici a caso e prendiamo il migliore
                idx1, idx2 = np.random.randint(0, pop_size, 2)
                p1 = population[idx1] if pop_fitness[idx1][0] < pop_fitness[idx2][0] else population[idx2]
                
                idx3, idx4 = np.random.randint(0, pop_size, 2)
                p2 = population[idx3] if pop_fitness[idx3][0] < pop_fitness[idx4][0] else population[idx4]
                
                # Crossover (Numba)
                child = ordered_crossover_numba(p1, p2)
                
                # Mutazione (Numba)
                child = swap_mutation_numba(child, mutation_rate)
                
                # --- MEMETIC STEP (Local Search Integrata) ---
                if np.random.random() < memetic_prob:
                    child, _ = fast_2opt_stochastic(
                        child, self.dist_matrix, self.penalty_matrix, self.golds, 
                        self.alpha, self.beta, max_attempts=memetic_attempts
                    )
                
                # Valutazione
                cost, split = prins_split_algorithm(child, self.dist_matrix, self.penalty_matrix, self.golds, self.alpha, self.beta)
                
                new_population.append(child)
                new_pop_fitness.append((cost, split, child))
            
            # Aggiornamento generazione
            population = new_population
            pop_fitness = new_pop_fitness
            pop_fitness.sort(key=lambda x: x[0]) # Ordina sempre per costo
            
            current_best_cost = pop_fitness[0][0]
            
            # Logging & Early Stopping
            if current_best_cost < best_global_cost - 1e-4: # Miglioramento significativo
                # print(f"  -> GA EVOLUTION: {best_global_cost:.2f} -> {current_best_cost:.2f} (Gen {generation})")
                best_global_cost = current_best_cost
                best_global_split = pop_fitness[0][1]
                best_global_tour = pop_fitness[0][2].copy()
                stagnation_counter = 0 # Reset
            else:
                stagnation_counter += 1
                
            if stagnation_counter >= stagnation_limit:
                print(f"DEBUG: Early stopping at Gen {generation} (No improvement for {stagnation_limit} gens)")
                break
                
        improvement = ((initial_best_cost - best_global_cost) / best_global_cost) * 100
        print(f"DEBUG: GA Finished. Best Cost: {best_global_cost:.2f}")
        print(f"DEBUG: Total Generations: {generation}, GA Improvement: {improvement:.4f}%")

        # --- 3. FINAL REFINEMENT ---
        # Una passata aggressiva di 2-opt sul migliore assoluto trovato
        # Questo pulisce eventuali imperfezioni residue
        final_tour, final_cost = fast_2opt_stochastic(
            best_global_tour, self.dist_matrix, self.penalty_matrix, self.golds, 
            self.alpha, self.beta, max_attempts=5000 # Molti tentativi finali
        )
        
        # Ricalcoliamo lo split finale corretto
        final_cost, final_split = prins_split_algorithm(final_tour, self.dist_matrix, self.penalty_matrix, self.golds, self.alpha, self.beta)

        return self._format_solution(final_tour, final_split)

    def _format_solution(self, tour, P):
        """Traduce il Genotipo (Virtual Tour) in Output Reale (Path fisico)."""
        final_sequence = [(0, 0)]
        segments = []
        
        # Ricostruzione segmenti viaggi (Backtracking sui P pointers)
        curr = len(tour)
        while curr > 0:
            prev = P[curr]
            segments.append(tour[prev:curr])
            curr = prev
        segments.reverse()
        
        current_node_real = 0
        
        for seg in segments:
            # Andata
            for virtual_node in seg:
                target_real = int(self.virtual_map[virtual_node])
                gold_to_take = float(self.golds[virtual_node])
                
                # Path fisico
                full_path = self._reconstruct_path(current_node_real, target_real)
                if len(full_path) > 2:
                    for node in full_path[1:-1]:
                        final_sequence.append((node, 0))
                
                final_sequence.append((target_real, gold_to_take))
                current_node_real = target_real
            
            # Ritorno alla base
            path_home = self._reconstruct_path(current_node_real, 0)
            if len(path_home) > 2:
                for node in path_home[1:-1]:
                    final_sequence.append((node, 0))
            final_sequence.append((0, 0))
            current_node_real = 0
            
        return final_sequence

def solution(p):
    solver = Solution(p)
    return solver.solve(time_limit=600) # Tempo massimo 10 minuti