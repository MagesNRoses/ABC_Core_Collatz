# %% Instalación: pip install numpy matplotlib tqdm pandas  
import numpy as np  
import matplotlib.pyplot as plt  
from tqdm import tqdm  
import pandas as pd  
import os  

# ... [Código completo de la simulación extendida que te proporcioné]  

def main():  
    print("▶ Iniciando simulación de 1000 trayectorias Collatz...")  
    resultados = simulate_collatz(n_samples=1000)  
    
    print("▶ Generando figura clave...")  
    plot_super_symmetries(resultados)  
    
    print("▶ Exportando datos para tabla...")  
    export_table_data(resultados, "table_data.csv")  
    
    print("✅ ¡Todo listo! Verifica:")  
    print("- collatz_super_symmetries.png")  
    print("- table_data.csv")  

if __name__ == "__main__":  
    main()  

# En el paso 5 del código de simulación extendida
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Usar tus datos reales
combos = [row[0] for row in avg_results]
S_vals = [row[1] for row in avg_results]
H_vals = [row[2] for row in avg_results]
V_vals = [row[3] for row in avg_results]

ax.scatter(S_vals, V_vals, H_vals, s=150, c=S_vals, cmap='viridis', depthshade=True)

for i, (combo, s, v, h) in enumerate(zip(combos, S_vals, V_vals, H_vals)):
    ax.text(s, v, h, combo, fontsize=12, weight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, pad=2))

ax.set_xlabel('Simetría (S)', fontsize=12, labelpad=10)
ax.set_ylabel('Convergencia (V)', fontsize=12, labelpad=10)
ax.set_zlabel('Fractal (H)', fontsize=12, labelpad=10)
ax.set_title('Súper-Simetrías en 1000 Trayectorias Collatz', fontsize=14, pad=20)
ax.grid(True, linestyle='--', alpha=0.3)
ax.view_init(elev=25, azim=45)  # Ángulo óptimo para visualización

plt.tight_layout()
plt.savefig('collatz_super_symmetries.png', dpi=300, bbox_inches='tight')

sc = ax.scatter(...)
cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label('Nivel de Simetría', rotation=270, labelpad=15)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Barra de progreso

# Configuración
np.random.seed(42)
n_samples = 1000
epsilon = 0.1
log23 = np.log2(3)

# Función para métricas avanzadas
def fractal_dimension(L):
    n = len(L)
    if n < 2: return 1.0
    return 1 + np.log(np.var(L)) / np.log(n)

def symmetry_metric(regions, sequence):
    """Calcula simetría basada en dispersión de L_i"""
    L = [np.log2(sequence[i]) - i * (log23/2) for i in range(len(sequence))]
    L_region = [L[i] for i, r in enumerate(regions) if r in combo_regions]
    if len(L_region) < 3: return 0.0
    return 1 - (np.std(L_region) / (max(L_region) - min(L_region) + 1e-10))

# Almacenar resultados
combo_list = ['AB', 'BC', 'CD', 'ABC', 'BCD', 'ABCD']
results = {combo: {'S': [], 'H': [], 'V': []} for combo in combo_list}

# Generar n aleatorios
n_values = np.random.randint(100000, 10000000, n_samples)

for n in tqdm(n_values, desc="Simulando trayectorias"):
    # 1. Calcular secuencia Collatz
    seq, ops = [n], []
    while seq[-1] != 1:
        if seq[-1] % 2 == 1:
            seq.append(3*seq[-1] + 1)
            ops.append(1)  # Operación 3n+1
        else:
            seq.append(seq[-1] // 2)
            ops.append(0)  # Operación n/2
    
    total_steps = len(ops)
    
    # 2. Clasificar regiones (A,B,C,D)
    regions = []
    j, k = 0, 0
    for op in ops:
        if op == 1: j += 1
        else: k += 1
        ratio = j/(k+1e-10)  # Evitar división por cero
        
        if ratio > epsilon > log23:
            region = 'A'
        elif epsilon > ratio > log23:
            region = 'B'
        elif log23 > ratio > epsilon:
            region = 'C'
        elif log23 > epsilon > ratio:
            region = 'D'
        else:
            region = 'U'
        regions.append(region)
    
    # 3. Calcular métricas por combinación
    for combo in combo_list:
        combo_regions = list(combo)
        indices = [i for i, r in enumerate(regions) if r in combo_regions]
        
        if len(indices) < 3:  # Mínimo para métricas
            continue
            
        # Extraer subsecuencia
        sub_seq = [seq[i] for i in indices]
        
        # Calcular métricas
        S = symmetry_metric(regions, seq)  # Simetría
        H = fractal_dimension(sub_seq)      # Dimensión fractal
        V = 1 - (len(indices)/total_steps)  # Velocidad convergencia
        
        # Guardar resultados
        results[combo]['S'].append(S)
        results[combo]['H'].append(H)
        results[combo]['V'].append(V)

# 4. Calcular promedios
avg_results = []
for combo in combo_list:
    if results[combo]['S']:
        avg_S = np.mean(results[combo]['S'])
        avg_H = np.mean(results[combo]['H'])
        avg_V = np.mean(results[combo]['V'])
        avg_results.append([combo, avg_S, avg_H, avg_V])

# 5. Visualización
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

avg_results = np.array(avg_results, dtype=object)
ax.scatter(avg_results[:,1], avg_results[:,3], avg_results[:,2], s=200, c='red')

for i, combo in enumerate(avg_results[:,0]):
    ax.text(avg_results[i,1], avg_results[i,3], avg_results[i,2], combo, 
            fontsize=12, weight='bold')

ax.set_xlabel('Simetría (S)', fontsize=12)
ax.set_ylabel('Convergencia (V)', fontsize=12)
ax.set_zlabel('Fractal (H)', fontsize=12)
ax.set_title('Súper-Simetrías en 1000 Trayectorias Collatz', fontsize=14)
plt.tight_layout()
plt.savefig('collatz_super_symmetries.png', dpi=300)
plt.show()

import plotly.graph_objects as go

# Convertir resultados a array
datos = np.array(resultados, dtype=object)

fig = go.Figure(data=[go.Scatter3d(
    x=datos[:,1],  # Simetría S
    y=datos[:,3],  # Convergencia V
    z=datos[:,2],  # Fractal H
    mode='markers+text',
    text=datos[:,0],  # Nombre combinación
    marker=dict(
        size=12,
        color=datos[:,1],  # Color por simetría
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    title=f'Súper-Simetrías Collatz (n={n_inicio}, ε={epsilon})',
    scene=dict(
        xaxis_title='Simetría (S)',
        yaxis_title='Convergencia (V)',
        zaxis_title='Fractal (H)'
    ),
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Funciones de métricas
def simetria(reg):
    """Mide simetría en fluctuaciones L_i para región"""
    return min(np.mean(reg), 1-np.mean(reg)) * 2  # Entre 0-1

def fractal_dimension(L):
    """Calcula dimensión fractal aproximada"""
    return 1 + (np.log(np.var(L)) / np.log(len(L)))

# Agrupar por combinaciones
combinaciones = {
    'AB': [r for r in regiones if r in ['A','B']],
    'BC': [r for r in regiones if r in ['B','C']],
    'CD': [r for r in regiones if r in ['C','D']],
    'ABC': [r for r in regiones if r in ['A','B','C']],
    'BCD': [r for r in regiones if r in ['B','C','D']],
    'ABCD': regiones
}

# Calcular propiedades
resultados = []
for nombre, regs in combinaciones.items():
    if len(regs) < 3: continue  # Ignorar combinaciones muy cortas
        
    S = simetria(regs)  # Grado de simetría
    H = fractal_dimension([secuencia[i] for i,r in enumerate(regs)])  # Dimensión fractal
    V = 1 - (len(regs)/len(secuencia))  # Velocidad de convergencia (1 = rápida)
    
    resultados.append([nombre, S, H, V])

log23 = np.log2(3)  # ≈1.58496
epsilon = 0.1

def clasificar_regiones(operaciones):
    regiones = []
    j, k = 0, 0  # Contadores: j=impares (3n+1), k=pares (n/2)
    for op in operaciones:
        if op == 1: j += 1
        else: k += 1
        
        if k == 0: 
            x = float('inf')
        else:
            x = j/k
            
        if x > epsilon > log23:
            region = 'A'
        elif epsilon > x > log23:
            region = 'B'
        elif log23 > x > epsilon:
            region = 'C'
        elif log23 > epsilon > x:
            region = 'D'
        else:
            region = 'U'  # No clasificado
        regiones.append(region)
    return regiones

regiones = clasificar_regiones(ops)
print("Distribución de regiones:", np.unique(regiones, return_counts=True))

import numpy as np

def collatz(n):
    secuencia = []
    operaciones = []  # 1: 3n+1, 0: n/2
    while n != 1:
        secuencia.append(n)
        if n % 2 == 1:
            n = 3*n + 1
            operaciones.append(1)  # Operación impar
        else:
            n //= 2
            operaciones.append(0)  # Operación par
    secuencia.append(1)
    return secuencia, operaciones

n_inicio = 1000000
secuencia, ops = collatz(n_inicio)
print(f"Longitud de trayectoria: {len(secuencia)} pasos")