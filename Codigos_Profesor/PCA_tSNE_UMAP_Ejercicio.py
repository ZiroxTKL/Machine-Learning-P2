import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
import time
import matplotlib.pyplot as plt

# ── 1. Cargar datos desde CSV ──────────────────────────────────────
df = pd.read_csv('toro_10d_200.csv')
print(f"Shape: {df.shape}")
print(df.head())

# Separar features y etiquetas
X = df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']].values
y = df['clase'].values

# Estandarizar (IMPORTANTE antes de PCA, t-SNE y UMAP)
X_sc = StandardScaler().fit_transform(X)

# ── 2. PCA a 2D ───────────────────────────────────────────────────
# TODO: aplicar PCA con n_components=2
# Z_pca = ...
# var_ret = ...  # varianza retenida en %

# ── 3. t-SNE a 2D ─────────────────────────────────────────────────
# TODO: aplicar t-SNE con perplexity=30, max_iter=1000, init='pca'
# Z_tsne = ...

# ── 4. UMAP a 2D ──────────────────────────────────────────────────
# TODO: aplicar UMAP con n_neighbors=15, min_dist=0.1
# Z_umap = ...

# ── 5. Visualización: 3 métodos lado a lado ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                         num="Toro 10D — PCA vs t-SNE vs UMAP")
for ax, (Z, titulo) in zip(axes, [
    (Z_pca,  f'PCA 2D  (var={var_ret:.0f}%)'),
    (Z_tsne, f't-SNE 2D'),
    (Z_umap, f'UMAP 2D'),
]):
    sc = ax.scatter(Z[:,0], Z[:,1], c=y, cmap='tab10', s=20, alpha=0.85)
    plt.colorbar(sc, ax=ax, label='Clase')
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Dim 1', fontsize=12)
    ax.set_ylabel('Dim 2', fontsize=12)

plt.suptitle('Toro 10D — reducción de dimensionalidad',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
