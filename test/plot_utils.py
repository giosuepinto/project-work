import matplotlib.pyplot as plt
import numpy as np
import os

def plot_single_comparison(baseline, solver, scenario_name, output_path):
    """
    Genera un grafico a barre semplice per il confronto singolo.
    Usato da benchmark_single.py
    """
    plt.figure(figsize=(7, 6))
    
    labels = ['Baseline', 'My Solution']
    values = [baseline, solver]
    colors = ['#808080', '#2ca02c']  # Grigio vs Verde
    
    # Creazione barre
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Aggiunta etichette valori
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    multiplier = baseline / solver if solver > 0 else 1.0
    
    plt.title(f"Scenario: {scenario_name}\nImprovement: {multiplier:.2f}x", fontsize=14, fontweight='bold')
    plt.ylabel('Total Cost')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graph saved in: {output_path}")

def plot_massive_results(results_dict, N, output_folder):
    """
    Genera il grafico complesso con tutti gli scenari per un dato N.
    Usato da benchmark_massive.py.
    Ricrea lo stile verde/blu/rosso raggruppato.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    scenarios = list(results_dict.keys())
    multipliers = [d['improvement_x'] for d in results_dict.values()]
    groups = [d['group'] for d in results_dict.values()]
    
    # Assegnazione Colori in base al Gruppo
    colors = []
    for g in groups:
        if 'TSP-like' in g:
            colors.append('#50C878')  # Verde Smeraldo (Geometria pura)
        elif 'Linear' in g:
            colors.append('#5DADE2')  # Blu (Stabile)
        elif 'Concave' in g:
            colors.append('#58D68D')  # Verde Chiaro
        elif 'Convex' in g:
            colors.append('#2ECC71')  # Verde Acceso (Split Delivery power)
        else:
            colors.append('#808080')  # Grigio default

    # Colora di rosso se peggiora (moltiplicatore < 0.99)
    final_colors = []
    for val, col in zip(multipliers, colors):
        if val < 0.99:
            final_colors.append('#E74C3C') # Rosso
        else:
            final_colors.append(col)

    x = np.arange(len(scenarios))
    width = 0.8
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(x, multipliers, width, color=final_colors, edgecolor='black', alpha=0.8)

    # Linea di parità (1.0x)
    plt.axhline(y=1.0, color='black', linewidth=2, linestyle='-')

    # Scala logaritmica se i valori esplodono (es. TSP-like > 10x)
    if max(multipliers) > 10:
        plt.yscale('log')
        plt.yticks([0.8, 1, 1.5, 2, 5, 10, 20, 50], ['0.8', '1.0', '1.5', '2.0', '5.0', '10.0', '20.0', '50.0'])
    else:
        plt.ylim(bottom=0.8, top=max(multipliers) * 1.15)

    # Etichette valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                 f'{height:.2f}x',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

    # Configurazione Assi
    plt.title(f'Performance Multiplier - N = {N} Cities\n(Higher is Better. 1.0x = Tie with Baseline)', fontsize=16, fontweight='bold')
    plt.ylabel('Performance Multiplier (x times better)', fontsize=12)
    plt.xticks(x, scenarios, rotation=45, ha="right", fontsize=10)
    plt.grid(axis='y', which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    
    filename = f"chart_N{N}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Cumulative Graph Generated: {filepath}")