import numpy as np
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.preprocessing import StandardScaler

import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Caso 1: Swiss Roll — variedad 2D enrollada en R^3
# PCA lo aplasta; t-SNE y UMAP lo "desenrollan"
X_roll, y_roll = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

# X_roll tiene 3 dimensiones con estructura de espiral no lineal

# Caso 2: Moons en alta dimension — estructura no lineal incrustada en Rd
# Generamos moons en 2D y los proyectamos a 30 dimensiones con ruido
X_2d, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
noise_dims  = np.random.RandomState(42).randn(1000, 28) * 0.3
X_moons     = np.hstack([X_2d, noise_dims])   # shape (1000, 30)

# Trabajamos con el Swiss Roll para el ejemplo principal
X, y = X_roll, y_roll.astype(int) % 4   # 4 clases artificiales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Dataset: {X.shape}  |  Clases: {np.unique(y)}")

# PCA completo — scree plot
pca_full = PCA().fit(X_scaled)
varianza_acumulada = pca_full.explained_variance_ratio_.cumsum()
k_95 = np.argmax(varianza_acumulada >= 0.95) + 1

# PCA a 2D — medimos tiempo
t0 = time.time()
pca_2d  = PCA(n_components=2)
Z_pca   = pca_2d.fit_transform(X_scaled)
t_pca   = time.time() - t0
var_ret = pca_2d.explained_variance_ratio_.sum() * 100
print(f"Tiempo PCA: {t_pca:.4f}s | Varianza 2D: {var_ret:.1f}%")

# Visualización lado a lado: 3D original vs PCA 2D
fig = plt.figure(figsize=(14, 5), num="Swiss Roll: Original vs PCA")
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='tab10', s=8)
ax1.set_title('Swiss Roll original (3D)', fontsize=16)

ax2 = fig.add_subplot(122)
sc = ax2.scatter(Z_pca[:,0], Z_pca[:,1], c=y, cmap='tab10', s=8)
plt.colorbar(sc, ax=ax2, label='Clase')
ax2.set_title(f'PCA 2D — Varianza: {var_ret:.1f}%', fontsize=16)
ax2.set_xlabel('PC1', fontsize=14); ax2.set_ylabel('PC2', fontsize=14)
plt.tight_layout()
#plt.show()   # Observar: las clases se MEZCLAN en la proyeccion PCA

from sklearn.manifold import TSNE

# t-SNE sobre los datos estandarizados
t0 = time.time()
tsne = TSNE(
    n_components=2,
    perplexity=30,    # vecindario efectivo ~30 puntos
    #n_iter=1000,
    max_iter=1000, 
    random_state=42,
    init='pca'        # inicializacion con PCA = mas estable y rapido
)
Z_tsne  = tsne.fit_transform(X_scaled)
t_tsne  = time.time() - t0
print(f"Tiempo t-SNE: {t_tsne:.2f} s")
# IMPORTANTE: t-SNE NO tiene explained_variance_ratio_

# Comparacion visual PCA vs t-SNE
fig, axes = plt.subplots(1, 2, figsize=(14, 5), num="Comparación: PCA vs t-SNE")
for ax, Z, title, c_label in [
    (axes[0], Z_pca,  f'PCA 2D (varianza={var_ret:.0f}%)', 'PC'),
    (axes[1], Z_tsne, f't-SNE 2D (t={t_tsne:.1f}s)', 't-SNE'),
]:
    sc = ax.scatter(Z[:,0], Z[:,1], c=y, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Clase')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'{c_label} 1', fontsize=14)
    ax.set_ylabel(f'{c_label} 2', fontsize=14)
plt.tight_layout()
#plt.show()   # t-SNE separa las clases; PCA las mezcla

# pip install umap-learn
import umap

t0 = time.time()
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,   # cuantos vecinos considera cada punto
    min_dist=0.1,     # compacidad de los clusters en 2D
    random_state=42
)
Z_umap = reducer.fit_transform(X_scaled)
t_umap = time.time() - t0
print(f"Tiempo UMAP: {t_umap:.2f} s  (comparar con t-SNE: {t_tsne:.2f} s)")

# UMAP puede proyectar datos NUEVOS (t-SNE no puede)
X_nuevo = scaler.transform(np.array([[0.5, 0.3, -0.2]]))
Z_nuevo = reducer.transform(X_nuevo)   # <- esto t-SNE no tiene
print(f"Punto nuevo proyectado: {Z_nuevo}")

# Comparacion final: 3 metodos
fig, axes = plt.subplots(1, 3, figsize=(18, 5),num="Comparación: PCA vs t-SNE vs UMAP" )
for ax, Z, title in [
    (axes[0], Z_pca,  f'PCA (var={var_ret:.0f}%, t={t_pca:.3f}s)'),
    (axes[1], Z_tsne, f't-SNE (t={t_tsne:.1f}s)'),
    (axes[2], Z_umap, f'UMAP (t={t_umap:.1f}s)'),
]:
    sc = ax.scatter(Z[:,0], Z[:,1], c=y, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Clase')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Dim 1', fontsize=14); ax.set_ylabel('Dim 2', fontsize=14)
plt.suptitle('Swiss Roll — PCA vs t-SNE vs UMAP', fontsize=17, fontweight='bold')
plt.tight_layout();
#plt.show()

import pandas as pd

# ── Tabla comparativa de los tres metodos ──────────────────────
resultados = {
    'Metodo':      ['PCA 2D',  't-SNE 2D', 'UMAP 2D'],
    'Tiempo (s)':  [round(t_pca,4), round(t_tsne,2), round(t_umap,2)],
    'Var. ret.':   [f'{var_ret:.1f}%', 'N/A', 'N/A'],
    'Estructura':  ['Lineal global', 'Local no lineal', 'Local + global parcial'],
    'transform()': ['Si', 'No', 'Si'],
}
df = pd.DataFrame(resultados)
print(df.to_string(index=False))

# ── Scree plot — cuantos componentes necesita PCA ──────────────
pca_all = PCA().fit(X_scaled)
var_cum = pca_all.explained_variance_ratio_.cumsum()
k95     = np.argmax(var_cum >= 0.95) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5),num="Swiss Roll - PCA vs t-SNE vs UMAP")
axes[0].plot(range(1, len(var_cum)+1), var_cum, 'o-',
             color='#003087', markersize=5, linewidth=2)
axes[0].axhline(0.95, color='red', linestyle='--', label='95%')
axes[0].axvline(k95, color='orange', linestyle='--', label=f'k={k95}')
axes[0].set_xlabel('Numero de componentes', fontsize=14)
axes[0].set_ylabel('Varianza acumulada', fontsize=14)
axes[0].set_title('Scree Plot', fontsize=16); axes[0].legend(fontsize=13)

axes[1].bar(['PCA', 't-SNE', 'UMAP'],
            [t_pca, t_tsne, t_umap],
            color=['#003087','#0055B3','#00A3E0'])
axes[1].set_ylabel('Tiempo (s)', fontsize=14)
axes[1].set_title('Comparacion de Tiempos', fontsize=16)
axes[1].tick_params(labelsize=13)
plt.tight_layout();
plt.show()
