import networkx as nx
import numpy as np
import scipy.sparse
from numba import jit

@jit(nopython=True)
def floyd_warshall_precompute(adj_matrix, num_nodes, beta):
    """
    Calcola APSS (All-Pairs Shortest Path) per Distanza e Penalità.
    D[i,j] = Distanza minima
    P[i,j] = Somma dei (d_k ** beta) lungo il percorso minimo
    """
    # Inizializzazione
    D = np.full((num_nodes, num_nodes), 1e15, dtype=np.float64)
    P = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    
    for i in range(num_nodes):
        D[i, i] = 0.0
        P[i, i] = 0.0
        
    # Riempimento dai dati di adiacenza
    # adj_matrix è passata come array denso dove 0 significa no-edge (eccetto diagonale)
    # Ma attenzione: 0 può essere una distanza valida? Assumiamo dist > 0 per edges.
    rows, cols = adj_matrix.shape
    for r in range(rows):
        for c in range(cols):
            w = adj_matrix[r, c]
            if w > 0 or (r != c and w == 0): # Check if edge exists (gestione sparse)
                # Qui assumiamo che adj_matrix contenga i pesi. 0 significa NO ARCO.
                # Se w > 0 c'è un arco.
                if w > 0:
                    D[r, c] = w
                    # Pre-calcolo del termine di penalità per questo singolo arco
                    if beta == 0:
                        P[r, c] = 1.0 # Conta hops se beta 0
                    else:
                        P[r, c] = w ** beta

    # Floyd-Warshall Core
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Rilassamento
                new_dist = D[i, k] + D[k, j]
                if new_dist < D[i, j] - 1e-9: # Tolleranza float
                    D[i, j] = new_dist
                    P[i, j] = P[i, k] + P[k, j]
                # Se le distanze sono uguali, preferiamo quello con penalità minore!
                # Questo ottimizza il path per il VRP
                elif abs(new_dist - D[i, j]) < 1e-9:
                    new_pen = P[i, k] + P[k, j]
                    if new_pen < P[i, j]:
                        P[i, j] = new_pen
                        
    return D, P

def process_graph_data(graph, alpha, beta):
    """
    Prepara le matrici per il solver.
    """
    num_original_cities = len(graph.nodes)
    original_golds = np.array([graph.nodes[i]['gold'] for i in range(num_original_cities)], dtype=np.float64)
    
    # 1. Calcolo Matrici Reali (Exact Cost)
    # Per N piccolo/medio (<500), FW in Numba è istantaneo.
    # Per N > 500 potrebbe volerci 1-2 secondi, accettabile.
    adj_dense = nx.to_numpy_array(graph, weight='dist')
    
    # Calcolo ESATTO (lento all'avvio ma preciso)
    dist_matrix_orig, penalty_matrix_orig = floyd_warshall_precompute(adj_dense, num_original_cities, beta)
    adj_sparse = nx.to_scipy_sparse_array(graph, weight='dist')
    _, predecessors = scipy.sparse.csgraph.shortest_path(adj_sparse, directed=False, return_predecessors=True)

            
    # --- VIRTUAL NODES STRATEGY ---
    # (Logica identica a prima, ma proiettiamo ANCHE la penalty_matrix)
    
    virtual_map = []
    virtual_golds = []
    
    avg_gold = np.mean(original_golds[1:]) if num_original_cities > 1 else 0
    num_edges = graph.number_of_edges()
    max_edges = (num_original_cities * (num_original_cities - 1)) / 2
    density = num_edges / max_edges if max_edges > 0 else 0
    
    split_factor = 1
    should_split = False
    if beta >= 2.0: should_split = True
    elif beta > 1.5 and density > 0.25: should_split = True
        
    for i in range(num_original_cities):
        g = original_golds[i]
        if i == 0:
            virtual_map.append(0)
            virtual_golds.append(0.0)
            continue
            
        num_splits = 1
        if should_split and g > avg_gold * 0.5:
             ratio = g / avg_gold
             if beta > 2.2:
                 if ratio > 2.0: num_splits = 3
                 elif ratio > 1.0: num_splits = 2
             else:
                 if ratio > 1.5: num_splits = 2
        
        chunk_gold = g / num_splits
        for _ in range(num_splits):
            virtual_map.append(i)
            virtual_golds.append(chunk_gold)
            
    v_map_arr = np.array(virtual_map, dtype=np.int32)
    ix_grid = np.ix_(v_map_arr, v_map_arr)
    
    dist_matrix_virtual = dist_matrix_orig[ix_grid]
    penalty_matrix_virtual = penalty_matrix_orig[ix_grid]
    
    return {
        'dist_matrix': dist_matrix_virtual,
        'penalty_matrix': penalty_matrix_virtual, # NEW
        'golds': np.array(virtual_golds, dtype=np.float64),
        'virtual_map': v_map_arr,
        'predecessors': predecessors,
        'original_num': num_original_cities
    }