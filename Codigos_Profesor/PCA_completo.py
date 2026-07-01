"""
=============================================================
  PCA - Implementación basica
  CS3061 - Machine Learning | UTEC
  Docente: Manuel Eduardo Loaiza Fernandez
=============================================================

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # dataset de ejemplo

# ============================================================
# 1. DATOS DE EJEMPLO
# ============================================================
data = load_iris()
X = data.data  # shape (150, 4) — 150 muestras, 4 features

# ============================================================
# 2. ESTANDARIZAR (obligatorio antes de PCA)
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. PCA COMPLETO para encontrar M óptimo PRIMERO
# ============================================================
pca_full = PCA()  # sin especificar n_components → calcula todos
pca_full.fit(X_scaled)

# Varianza acumulada
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar M que explique 95% de varianza
M = np.argmax(cumsum >= 0.95) + 1
print(f"Componentes para 95% de varianza: {M}")

# Graficar varianza acumulada
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
plt.axvline(x=M, color='g', linestyle='--', label=f'M={M} componentes')
plt.legend()
plt.title('Varianza acumulada por componentes')
plt.grid(True)
plt.show()

# ============================================================
# 4. PCA CON M COMPONENTES ÓPTIMOS
# ============================================================
pca = PCA(n_components=M)  # ahora M ya está definido
pca.fit(X_scaled)

# Componentes principales
print("\n--- Resultados PCA ---")
print(f"Eigenvectores (U) shape: {pca.components_.shape}")
print(f"Eigenvectores (U):\n{pca.components_}")
print(f"\nEigenvalores (diagonal Σ):\n{pca.explained_variance_}")
print(f"\n% Varianza explicada por componente:\n{pca.explained_variance_ratio_}")
print(f"\nVarianza total explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# ============================================================
# 5. PROYECTAR Y RECONSTRUIR
# ============================================================
# Proyectar: D dimensiones → M dimensiones
Z = pca.transform(X_scaled)
print(f"\nShape original:    {X_scaled.shape}")
print(f"Shape proyectado:  {Z.shape}")

# Reconstruir: M dimensiones → D dimensiones
X_reconstructed = pca.inverse_transform(Z)
print(f"Shape reconstruido: {X_reconstructed.shape}")

# Error de reconstrucción
error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nError de reconstrucción (MSE): {error:.6f}")

# ============================================================
# 6. VISUALIZAR PROYECCIÓN 2D (si M >= 2)
# ============================================================
if M >= 2:
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], 
                         c=data.target, 
                         cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Proyección PCA — {M} componentes')
    plt.colorbar(scatter, label='Clase')
    plt.grid(True)
    plt.show()
