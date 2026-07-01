"""
=============================================================
  K-Nearest Neighbors - Implementación desde cero
  CS3061 - Machine Learning | UTEC
  Docente: Manuel Eduardo Loaiza Fernandez
=============================================================

ESTRUCTURA DEL CÓDIGO:
  1. Implementación KNN desde cero
  2. Funciones auxiliares (distancia, fronteras, accuracy)
  3. CASO 1 - Dataset sintético: K=sqrt(n) es óptimo
  4. CASO 2 - Dataset sintético: K muy grande (underfitting)
  5. CASO 3 - Dataset sintético: K muy pequeño (overfitting)
  6. CASO 4 - Dataset real Iris: alumno busca K óptimo
  7. CASO 5 - Dataset real Wine: alumno busca K óptimo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────────────────────
#  SEMILLA PARA REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────
np.random.seed(42) # What happen if uncomment this line ? 


# =============================================================
#  1. IMPLEMENTACIÓN KNN DESDE CERO
# =============================================================

def distancia_euclidiana(x1, x2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    """
    Implementación de K-Nearest Neighbors desde cero.
    Sin uso de sklearn para el clasificador.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Entrenar = simplemente almacenar los datos."""
        self.X_train = X
        self.y_train = y

    def predecir_uno(self, x):
        """Predice la clase de un único punto."""
        # Calcular distancia a todos los puntos de entrenamiento
        distancias = [distancia_euclidiana(x, x_train)
                      for x_train in self.X_train]

        # Obtener índices de los k vecinos más cercanos
        indices_k = np.argsort(distancias)[:self.k]

        # Obtener etiquetas de los k vecinos
        etiquetas_k = [self.y_train[i] for i in indices_k]

        # Votación mayoritaria
        conteo = Counter(etiquetas_k)
        return conteo.most_common(1)[0][0]

    def predecir(self, X):
        """Predice las clases de múltiples puntos."""
        return np.array([self.predecir_uno(x) for x in X])

    def accuracy(self, X, y):
        """Calcula la precisión del modelo."""
        predicciones = self.predecir(X)
        return np.mean(predicciones == y)


# =============================================================
#  2. FUNCIONES AUXILIARES
# =============================================================

def train_test_split_manual(X, y, test_size=0.3):
    """División manual train/test sin sklearn."""
    n = len(X)
    n_test = int(n * test_size)
    indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalizar(X_train, X_test):
    """Normalización min-max usando solo estadísticas de train."""
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    rango = X_max - X_min
    rango[rango == 0] = 1  # evitar división por cero
    X_train_norm = (X_train - X_min) / rango
    X_test_norm = (X_test - X_min) / rango
    return X_train_norm, X_test_norm


def graficar_frontera(ax, modelo, X, y, titulo, k, acc,
                      colores_mapa, colores_puntos):
    """
    Grafica la frontera de decisión de un modelo KNN
    sobre un espacio 2D usando solo las primeras 2 features.
    """
    h = 0.05  # resolución de la malla
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    malla = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predecir(malla)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=colores_mapa)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y,
                         cmap=colores_mapa, edgecolors='k',
                         linewidths=0.5, s=40)
    ax.set_title(f"{titulo}\nK={k} | Accuracy={acc:.2%}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def graficar_accuracy_vs_k(ax, X_train, y_train, X_test, y_test,
                           k_range, k_sqrt, titulo):
    """
    Grafica la curva de accuracy vs K para train y test,
    marcando K=sqrt(n) como referencia.
    """
    acc_train = []
    acc_test = []

    for k in k_range:
        modelo = KNN(k=k)
        modelo.fit(X_train, y_train)
        acc_train.append(modelo.accuracy(X_train, y_train))
        acc_test.append(modelo.accuracy(X_test, y_test))

    ax.plot(k_range, acc_train, 'b-o', markersize=4,
            label='Train accuracy', linewidth=1.5)
    ax.plot(k_range, acc_test, 'r-s', markersize=4,
            label='Test accuracy', linewidth=1.5)
    ax.axvline(x=k_sqrt, color='green', linestyle='--',
               linewidth=2, label=f'K=√n={k_sqrt}')
    ax.set_xlabel("Valor de K")
    ax.set_ylabel("Accuracy")
    ax.set_title(titulo, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)


# =============================================================
#  3. CASO 1 — Dataset sintético: K=sqrt(n) es ÓPTIMO
# =============================================================

def caso1_k_optimo():
    """
    Datos sintéticos bien separados con algo de ruido.
    sqrt(n) es una buena elección de K.
    """
    print("\n" + "="*60)
    print("  CASO 1: K=sqrt(n) es óptimo")
    print("="*60)

    # Generar datos: 2 clases, 2 features, 100 muestras
    n = 100
    X_clase0 = np.random.randn(n // 2, 2) + np.array([-1.5, -1.5])
    X_clase1 = np.random.randn(n // 2, 2) + np.array([1.5, 1.5])
    X = np.vstack([X_clase0, X_clase1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    X_train, X_test = normalizar(X_train, X_test)

    n_train = len(X_train)
    k_sqrt = max(1, int(np.sqrt(n_train)))
    print(f"  n_train = {n_train} | K = sqrt({n_train}) = {k_sqrt}")

    # Evaluar K pequeño, sqrt(n) y K grande
    k_pequeno = 1
    k_grande = 30
    k_range = range(1, 31)

    colores_mapa = ListedColormap(['#AADDFF', '#FFAAAA'])

    #fig = plt.figure(figsize=(16, 10))
    fig = plt.figure(
    figsize=(16,10),
    constrained_layout=True )
    
    fig.suptitle("CASO 1: Dataset Sintético — K=√n es óptimo",
                 fontsize=14, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Fronteras de decisión para 3 valores de K
    for idx, k_val in enumerate([k_pequeno, k_sqrt, k_grande]):
        modelo = KNN(k=k_val)
        modelo.fit(X_train, y_train)
        acc = modelo.accuracy(X_test, y_test)

        if k_val == k_pequeno:
            etiqueta = f"K={k_val} (muy pequeño)"
        elif k_val == k_sqrt:
            etiqueta = f"K={k_val} = √n (óptimo)"
        else:
            etiqueta = f"K={k_val} (muy grande)"

        ax = fig.add_subplot(gs[0, idx])
        graficar_frontera(ax, modelo, X_test, y_test,
                          etiqueta, k_val, acc,
                          colores_mapa, ['blue', 'red'])
        print(f"  K={k_val:2d} → Test Accuracy: {acc:.2%}")

    # Curva accuracy vs K
    ax_curve = fig.add_subplot(gs[1, :])
    graficar_accuracy_vs_k(ax_curve, X_train, y_train,
                           X_test, y_test, k_range, k_sqrt,
                           "Accuracy vs K — Caso 1")

    #plt.tight_layout()
    #fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('caso1_k_optimo.png',
                dpi=120, bbox_inches='tight')

    #plt.show(block=True)      # mostrar ventana
    #plt.close()
    print(f"\n  ✅ Gráfico guardado: caso1_k_optimo.png")


# =============================================================
#  4. CASO 2 — K MUY GRANDE (underfitting)
# =============================================================

def caso2_k_grande():
    """
    Datos con frontera no lineal compleja.
    K muy grande produce underfitting — frontera demasiado suave.
    """
    print("\n" + "="*60)
    print("  CASO 2: K muy grande → Underfitting")
    print("="*60)

    # Generar datos en forma de lunas (no linealmente separables)
    n = 200
    angulos0 = np.linspace(0, np.pi, n // 2)
    angulos1 = np.linspace(np.pi, 2 * np.pi, n // 2)

    X_clase0 = np.c_[np.cos(angulos0), np.sin(angulos0)]
    X_clase1 = np.c_[np.cos(angulos1) + 0.5,
                     np.sin(angulos1) - 0.3]

    ruido = 0.15
    X_clase0 += np.random.randn(n // 2, 2) * ruido
    X_clase1 += np.random.randn(n // 2, 2) * ruido

    X = np.vstack([X_clase0, X_clase1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    X_train, X_test = normalizar(X_train, X_test)

    n_train = len(X_train)
    k_sqrt = max(1, int(np.sqrt(n_train)))
    k_grande = 80  # deliberadamente muy grande
    k_range = range(1, 91)

    print(f"  n_train = {n_train} | K=√n = {k_sqrt} | K grande = {k_grande}")

    colores_mapa = ListedColormap(['#AADDFF', '#FFAAAA'])

    #fig = plt.figure(figsize=(16, 10))
    fig = plt.figure(
    figsize=(16,10),
    constrained_layout=True )
    
    fig.suptitle("CASO 2: K muy grande → Underfitting (frontera demasiado suave)",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    for idx, k_val in enumerate([k_sqrt, k_grande]):
        modelo = KNN(k=k_val)
        modelo.fit(X_train, y_train)
        acc = modelo.accuracy(X_test, y_test)

        etiqueta = (f"K={k_val} = √n (referencia)"
                    if k_val == k_sqrt
                    else f"K={k_val} (muy grande → underfitting)")

        ax = fig.add_subplot(gs[0, idx])
        graficar_frontera(ax, modelo, X_test, y_test,
                          etiqueta, k_val, acc,
                          colores_mapa, ['blue', 'red'])
        print(f"  K={k_val:2d} → Test Accuracy: {acc:.2%}")

    ax_curve = fig.add_subplot(gs[1, :])
    graficar_accuracy_vs_k(ax_curve, X_train, y_train,
                           X_test, y_test, k_range, k_sqrt,
                           "Accuracy vs K — Caso 2 (K grande = underfitting)")

    #plt.tight_layout()
    plt.savefig('caso2_k_grande.png',
                dpi=120, bbox_inches='tight')

    #plt.show(block=True)      # mostrar ventana
    #plt.close()
    print(f"\n  ✅ Gráfico guardado: caso2_k_grande.png")


# =============================================================
#  5. CASO 3 — K MUY PEQUEÑO (overfitting)
# =============================================================

def caso3_k_pequeno():
    """
    Datos con ruido fuerte.
    K=1 produce overfitting — memoriza el ruido.
    """
    print("\n" + "="*60)
    print("  CASO 3: K muy pequeño → Overfitting")
    print("="*60)

    # 3 clases con mucho ruido
    n_por_clase = 60
    centros = [[-2, -2], [2, -2], [0, 2.5]]
    X_list, y_list = [], []

    for i, centro in enumerate(centros):
        X_c = np.random.randn(n_por_clase, 2) * 1.2 + centro
        X_list.append(X_c)
        y_list.append(np.full(n_por_clase, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Añadir ruido de etiquetas (10% mal etiquetados)
    n_ruido = int(0.10 * len(y))
    idx_ruido = np.random.choice(len(y), n_ruido, replace=False)
    y[idx_ruido] = np.random.randint(0, 3, n_ruido)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    X_train, X_test = normalizar(X_train, X_test)

    n_train = len(X_train)
    k_sqrt = max(1, int(np.sqrt(n_train)))
    k_pequeno = 1
    k_range = range(1, 31)

    print(f"  n_train = {n_train} | K=√n = {k_sqrt} | K pequeño = {k_pequeno}")

    colores_mapa = ListedColormap(['#AADDFF', '#FFAAAA', '#AAFFAA'])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CASO 3: K muy pequeño → Overfitting (memoriza el ruido)",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    for idx, k_val in enumerate([k_pequeno, k_sqrt]):
        modelo = KNN(k=k_val)
        modelo.fit(X_train, y_train)
        acc = modelo.accuracy(X_test, y_test)

        etiqueta = (f"K={k_val} (muy pequeño → overfitting)"
                    if k_val == k_pequeno
                    else f"K={k_val} = √n (referencia)")

        ax = fig.add_subplot(gs[0, idx])
        graficar_frontera(ax, modelo, X_test, y_test,
                          etiqueta, k_val, acc,
                          colores_mapa, ['blue', 'red', 'green'])
        print(f"  K={k_val:2d} → Test Accuracy: {acc:.2%}")

    ax_curve = fig.add_subplot(gs[1, :])
    graficar_accuracy_vs_k(ax_curve, X_train, y_train,
                           X_test, y_test, k_range, k_sqrt,
                           "Accuracy vs K — Caso 3 (K pequeño = overfitting)")

    #plt.tight_layout()
    plt.savefig('caso3_k_pequeno.png',
                dpi=120, bbox_inches='tight')
    #plt.close()
    print(f"\n  ✅ Gráfico guardado: caso3_k_pequeno.png")


# =============================================================
#  6. CASO 4 — Dataset real IRIS: alumno busca K óptimo
# =============================================================

def caso4_iris_buscar_k():
    """
    Dataset Iris real (cargado manualmente).
    El alumno debe encontrar el K óptimo explorando la curva.
    Solo se usan las primeras 2 features para visualización.
    """
    print("\n" + "="*60)
    print("  CASO 4: Dataset Iris — El alumno busca K óptimo")
    print("="*60)

    # Dataset Iris embebido directamente (150 muestras, 4 features)
    # Fuente: Fisher, 1936
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_full = iris.data        # 4 features
    X = iris.data[:, :2]     # primeras 2 features para visualización
    y = iris.target           # 3 clases

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.3)
    X_train, X_test = normalizar(X_train, X_test)

    n_train = len(X_train)
    k_sqrt = max(1, int(np.sqrt(n_train)))
    k_range = range(1, 31)

    print(f"  Dataset: Iris | n_train={n_train} | K=√n={k_sqrt}")
    print(f"  Features usadas para visualización: sepal length, sepal width")
    print(f"  Clases: {iris.target_names.tolist()}")
    print()

    # Evaluar todos los K y mostrar tabla
    print(f"  {'K':>4} | {'Train Acc':>10} | {'Test Acc':>10}")
    print(f"  {'-'*30}")
    mejor_k = 1
    mejor_acc = 0.0

    for k in k_range:
        modelo = KNN(k=k)
        modelo.fit(X_train, y_train)
        acc_tr = modelo.accuracy(X_train, y_train)
        acc_te = modelo.accuracy(X_test, y_test)
        marca = " ← √n" if k == k_sqrt else ""
        print(f"  K={k:2d} | Train={acc_tr:.2%} | Test={acc_te:.2%}{marca}")
        if acc_te > mejor_acc:
            mejor_acc = acc_te
            mejor_k = k

    print(f"\n  >>> K óptimo encontrado automáticamente: K={mejor_k}"
          f" (Test Acc={mejor_acc:.2%})")
    print(f"  >>> ¿Coincide con K=√n={k_sqrt}?",
          "Sí ✅" if mejor_k == k_sqrt else "No ❌ — ¡investiga por qué!")

    colores_mapa = ListedColormap(['#AADDFF', '#FFAAAA', '#AAFFAA'])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CASO 4: Dataset Iris — ¿Cuál es el K óptimo?",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    for idx, k_val in enumerate([k_sqrt - 2, k_sqrt, mejor_k]):
        k_val = max(1, k_val)
        modelo = KNN(k=k_val)
        modelo.fit(X_train, y_train)
        acc = modelo.accuracy(X_test, y_test)

        if k_val == mejor_k and k_val != k_sqrt:
            etiqueta = f"K={k_val} (óptimo encontrado)"
        elif k_val == k_sqrt:
            etiqueta = f"K={k_val} = √n"
        else:
            etiqueta = f"K={k_val} (menor a √n)"

        ax = fig.add_subplot(gs[0, idx])
        graficar_frontera(ax, modelo, X_test, y_test,
                          etiqueta, k_val, acc,
                          colores_mapa, ['blue', 'red', 'green'])

    ax_curve = fig.add_subplot(gs[1, :])
    graficar_accuracy_vs_k(ax_curve, X_train, y_train,
                           X_test, y_test, k_range, k_sqrt,
                           "Accuracy vs K — Caso 4: Iris\n"
                           "¿Dónde está el K óptimo?")

    # Anotación de tarea para el alumno
    ax_curve.annotate("¿Cuál es el mejor K?\n¡Analiza la curva!",
                      xy=(mejor_k, mejor_acc),
                      xytext=(mejor_k + 3, mejor_acc - 0.1),
                      arrowprops=dict(arrowstyle='->', color='purple'),
                      fontsize=9, color='purple', fontweight='bold')

    #plt.tight_layout()
    plt.savefig('caso4_iris.png',
                dpi=120, bbox_inches='tight')
    #plt.close()
    print(f"\n  ✅ Gráfico guardado: caso4_iris.png")


# =============================================================
#  7. CASO 5 — Dataset real WINE: alumno busca K óptimo
# =============================================================

def caso5_wine_buscar_k():
    """
    Dataset Wine real.
    Más complejo que Iris — el alumno debe buscar K con más cuidado.
    Solo se usan las primeras 2 features para visualización.
    """
    print("\n" + "="*60)
    print("  CASO 5: Dataset Wine — El alumno busca K óptimo")
    print("="*60)

    from sklearn.datasets import load_wine
    wine = load_wine()
    X = wine.data[:, :2]   # primeras 2 features para visualización
    y = wine.target         # 3 clases de vino

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.3)
    X_train, X_test = normalizar(X_train, X_test)

    n_train = len(X_train)
    k_sqrt = max(1, int(np.sqrt(n_train)))
    k_range = range(1, 31)

    print(f"  Dataset: Wine | n_train={n_train} | K=√n={k_sqrt}")
    print(f"  Features: alcohol, malic_acid")
    print(f"  Clases: {wine.target_names.tolist()}")
    print()

    print(f"  {'K':>4} | {'Train Acc':>10} | {'Test Acc':>10}")
    print(f"  {'-'*30}")
    mejor_k = 1
    mejor_acc = 0.0

    for k in k_range:
        modelo = KNN(k=k)
        modelo.fit(X_train, y_train)
        acc_tr = modelo.accuracy(X_train, y_train)
        acc_te = modelo.accuracy(X_test, y_test)
        marca = " ← √n" if k == k_sqrt else ""
        print(f"  K={k:2d} | Train={acc_tr:.2%} | Test={acc_te:.2%}{marca}")
        if acc_te > mejor_acc:
            mejor_acc = acc_te
            mejor_k = k

    print(f"\n  >>> K óptimo encontrado automáticamente: K={mejor_k}"
          f" (Test Acc={mejor_acc:.2%})")
    print(f"  >>> ¿Coincide con K=√n={k_sqrt}?",
          "Sí ✅" if mejor_k == k_sqrt else "No ❌ — ¡investiga por qué!")

    colores_mapa = ListedColormap(['#AADDFF', '#FFAAAA', '#AAFFAA'])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CASO 5: Dataset Wine — ¿Cuál es el K óptimo?",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    for idx, k_val in enumerate([k_sqrt - 2, k_sqrt, mejor_k]):
        k_val = max(1, k_val)
        modelo = KNN(k=k_val)
        modelo.fit(X_train, y_train)
        acc = modelo.accuracy(X_test, y_test)

        if k_val == mejor_k and k_val != k_sqrt:
            etiqueta = f"K={k_val} (óptimo encontrado)"
        elif k_val == k_sqrt:
            etiqueta = f"K={k_val} = √n"
        else:
            etiqueta = f"K={k_val} (menor a √n)"

        ax = fig.add_subplot(gs[0, idx])
        graficar_frontera(ax, modelo, X_test, y_test,
                          etiqueta, k_val, acc,
                          colores_mapa, ['blue', 'red', 'green'])

    ax_curve = fig.add_subplot(gs[1, :])
    graficar_accuracy_vs_k(ax_curve, X_train, y_train,
                           X_test, y_test, k_range, k_sqrt,
                           "Accuracy vs K — Caso 5: Wine\n"
                           "¿Dónde está el K óptimo?")

    ax_curve.annotate("¿Cuál es el mejor K?\n¡Analiza la curva!",
                      xy=(mejor_k, mejor_acc),
                      xytext=(mejor_k + 3, mejor_acc - 0.1),
                      arrowprops=dict(arrowstyle='->', color='purple'),
                      fontsize=9, color='purple', fontweight='bold')

    #plt.tight_layout()
    plt.savefig('caso5_wine.png',
                dpi=120, bbox_inches='tight')
    #plt.close()
    print(f"\n  ✅ Gráfico guardado: caso5_wine.png")


# =============================================================
#  MAIN — Ejecutar todos los casos
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  KNN DESDE CERO — CS3061 Machine Learning | UTEC")
    print("="*60)
    print("""
  CASOS INCLUIDOS:
  ─────────────────────────────────────────────────────
  Caso 1: Datos sintéticos → K=√n es óptimo
  Caso 2: Datos sintéticos → K muy grande (underfitting)
  Caso 3: Datos sintéticos → K muy pequeño (overfitting)
  Caso 4: Dataset Iris     → Alumno busca K óptimo
  Caso 5: Dataset Wine     → Alumno busca K óptimo
  ─────────────────────────────────────────────────────
    """)

    caso1_k_optimo()
    caso2_k_grande()
    caso3_k_pequeno()
    caso4_iris_buscar_k()
    caso5_wine_buscar_k()

    plt.show()

    print("\n" + "="*60)
    print("  ✅ Todos los casos completados.")
    print("  Archivos generados:")
    print("    → caso1_k_optimo.png")
    print("    → caso2_k_grande.png")
    print("    → caso3_k_pequeno.png")
    print("    → caso4_iris.png")
    print("    → caso5_wine.png")
    print("="*60 + "\n")
