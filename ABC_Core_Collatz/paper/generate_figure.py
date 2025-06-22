# generate_figure.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Datos fijos de tu investigación (no necesitas ejecutar simulaciones)
combinations = ['AB', 'BC', 'CD', 'ABC', 'ABCD']
S_values = [0.72, 0.81, 0.74, 0.92, 0.95]
H_values = [1.28, 1.39, 1.31, 1.60, 1.68]
V_values = [0.68, 0.72, 0.69, 0.87, 0.93]

# Crear la figura 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Configuración estética
ax.grid(True, linestyle='--', alpha=0.4)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Crear gráfico 3D con colores
scatter = ax.scatter(
    S_values, V_values, H_values, 
    s=200, 
    c=S_values, 
    cmap='viridis',
    depthshade=True
)

# Añadir etiquetas
for i, combo in enumerate(combinations):
    ax.text(
        S_values[i], V_values[i], H_values[i] + 0.02, 
        combo, 
        fontsize=14,
        weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, pad=3, edgecolor='none')
    )

# Ejes y título
ax.set_xlabel('Symmetry (S)', fontsize=12, labelpad=15)
ax.set_ylabel('Convergence (V)', fontsize=12, labelpad=15)
ax.set_zlabel('Fractal Dimension (H)', fontsize=12, labelpad=15)
ax.set_title('Super-Symmetries in Collatz Trajectories', fontsize=16, pad=20)

# Barra de color
cbar = fig.colorbar(scatter, pad=0.1)
cbar.set_label('Symmetry Level', rotation=270, labelpad=20)

# Ajustar vista
ax.view_init(elev=25, azim=45)

# Guardar figura
plt.tight_layout()
plt.savefig('collatz_super_symmetries.png', dpi=300, bbox_inches='tight')
print("¡Figura generada con éxito: collatz_super_symmetries.png")