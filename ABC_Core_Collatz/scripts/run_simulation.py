import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# --- Funciones esenciales ---
def collatz(n):
    seq, ops = [n], []
    current = n
    while current != 1:
        if current % 2 == 1:
            current = 3 * current + 1
            ops.append(1)
        else:
            current //= 2
            ops.append(0)
        seq.append(current)
    return seq, ops

def clasificar_regiones(operaciones):
    log23 = np.log2(3)
    epsilon = 0.1
    regions = []
    j, k = 0, 0
    for op in operaciones:
        if op == 1: j += 1
        else: k += 1
        ratio = j/(k + 1e-10)
        
        if ratio > epsilon > log23: region = 'A'
        elif epsilon > ratio > log23: region = 'B'
        elif log23 > ratio > epsilon: region = 'C'
        elif log23 > epsilon > ratio: region = 'D'
        else: region = 'U'
        regions.append(region)
    return regions

def symmetry_metric(L):
    """Calcula métrica de simetría para una secuencia L"""
    if len(L) < 3: return 0.0
    return 1 - (np.std(L) / (max(L) - min(L) + 1e-10))

def fractal_dimension(L):
    """Calcula dimensión fractal aproximada"""
    if len(L) < 2: return 1.0
    return 1 + np.log(np.var(L)) / np.log(len(L))

# --- Función principal ---
def main():
    np.random.seed(42)
    n_samples = 1000
    n_values = np.random.randint(100000, 10000000, n_samples)
    log23 = np.log2(3)
    epsilon = 0.1
    
    resultados = []
    combo_list = ['AB', 'BC', 'CD', 'ABC', 'ABCD']
    
    for n in tqdm(n_values, desc="Simulando trayectorias"):
        seq, ops = collatz(n)
        regions = clasificar_regiones(ops)
        total_steps = len(ops)
        
        L = [np.log2(seq[i]) - i * log23/2 for i in range(len(seq))]
        
        for combo in combo_list:
            combo_regions = list(combo)
            indices = [i for i, r in enumerate(regions) if r in combo_regions]
            
            if len(indices) < 3: continue
                
            sub_L = [L[i] for i in indices]
            S = symmetry_metric(sub_L)
            H = fractal_dimension([seq[i] for i in indices])
            V = 1 - (len(indices)/total_steps
            
            resultados.append({
                'combo': combo,
                'S': S,
                'H': H,
                'V': V,
                'n': n
            })
    
    # Exportar resultados
    df = pd.DataFrame(resultados)
    df_grouped = df.groupby('combo').agg({'S': ['mean', 'std'], 'H': ['mean', 'std'], 'V': ['mean', 'std']})
    df_grouped.to_csv('paper/table_data.csv')
    
    # Generar figura
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for combo in combo_list:
        subset = df[df['combo'] == combo]
        ax.scatter(subset['S'], subset['V'], subset['H'], label=combo, s=20)
    
    ax.set_xlabel('Simetría (S)')
    ax.set_ylabel('Convergencia (V)')
    ax.set_zlabel('Fractal (H)')
    ax.legend()
    plt.savefig('paper/collatz_super_symmetries.png', dpi=300)
    print("✅ Simulación completada y figuras generadas")

if __name__ == "__main__":
    main()