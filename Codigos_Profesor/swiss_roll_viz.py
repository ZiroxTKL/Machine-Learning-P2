import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 1.  Generación de datos sintéticos
# ─────────────────────────────────────────────

# Caso 1: Swiss Roll — variedad 2D enrollada en R^3
X_roll, y_roll = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

# Caso 2: Moons en alta dimensión
X_2d, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
noise_dims = np.random.RandomState(42).randn(1000, 28) * 0.3
X_moons    = np.hstack([X_2d, noise_dims])   # shape (1000, 30)

# Etiquetas de clase para Swiss Roll (4 clases artificiales)
X, y = X_roll, y_roll.astype(int) % 4
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Swiss Roll  — shape: {X.shape}  |  clases: {np.unique(y)}")
print(f"Moons HD    — shape: {X_moons.shape}")

# ─────────────────────────────────────────────
# 2.  PCA sobre Swiss Roll (3D → 2D)
# ─────────────────────────────────────────────
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_

# ─────────────────────────────────────────────
# 3.  Paleta y estilo
# ─────────────────────────────────────────────
COLORS      = ['#378ADD', '#E24B4A', '#1D9E75', '#EF9F27']
CLASS_NAMES = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']
CMAP        = plt.cm.Spectral            # para colorear por t continuo
ALPHA       = 0.70
PT_SIZE_3D  = 8
PT_SIZE_2D  = 10

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F8F8',
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})

# ─────────────────────────────────────────────
# 4.  Figura principal: Swiss Roll
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle('Swiss Roll — Visualización completa del dataset sintético',
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.40, wspace=0.35,
                       left=0.05, right=0.97,
                       top=0.93, bottom=0.06)

# ── 4a.  Vista 3D libre (coloreado por clase) ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
for cls, col, lbl in zip(range(4), COLORS, CLASS_NAMES):
    mask = y == cls
    ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                c=col, s=PT_SIZE_3D, alpha=ALPHA, label=lbl, depthshade=True)
ax1.set_title('Swiss Roll 3D — por clase', fontsize=11, pad=8)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend(loc='upper left', fontsize=8, markerscale=1.4)
ax1.view_init(elev=20, azim=45)

# ── 4b.  Vista 3D coloreada por valor continuo t (estructura real) ─────────
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
sc = ax2.scatter(X[:, 0], X[:, 1], X[:, 2],
                 c=y_roll, cmap=CMAP, s=PT_SIZE_3D, alpha=ALPHA)
plt.colorbar(sc, ax=ax2, shrink=0.55, pad=0.08, label='t (posición en espiral)')
ax2.set_title('Swiss Roll 3D — por t continuo', fontsize=11, pad=8)
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.view_init(elev=20, azim=120)

# ── 4c.  PCA 2D — resultado de la proyección lineal ────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for cls, col, lbl in zip(range(4), COLORS, CLASS_NAMES):
    mask = y == cls
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=col, s=PT_SIZE_2D, alpha=ALPHA, label=lbl, edgecolors='none')
ax3.set_title(
    f'Proyección PCA 2D\n'
    f'PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}  '
    f'Total={sum(var_explained):.1%}',
    fontsize=10)
ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2')
ax3.legend(fontsize=8, markerscale=1.4)
# Anotación sobre la mezcla de clases
ax3.text(0.02, 0.97,
         '⚠ PCA mezcla las clases\n(estructura no lineal perdida)',
         transform=ax3.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                   edgecolor='#EF9F27', alpha=0.9))

# ── 4d.  Vista superior del Swiss Roll (X vs Z) ────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for cls, col, lbl in zip(range(4), COLORS, CLASS_NAMES):
    mask = y == cls
    ax4.scatter(X[mask, 0], X[mask, 2],
                c=col, s=PT_SIZE_2D, alpha=ALPHA, label=lbl, edgecolors='none')
ax4.set_title('Vista superior (X vs Z) — espiral visible', fontsize=10)
ax4.set_xlabel('X'); ax4.set_ylabel('Z')
ax4.legend(fontsize=8, markerscale=1.4)

# ── 4e.  Vista lateral (X vs Y) ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for cls, col, lbl in zip(range(4), COLORS, CLASS_NAMES):
    mask = y == cls
    ax5.scatter(X[mask, 0], X[mask, 1],
                c=col, s=PT_SIZE_2D, alpha=ALPHA, label=lbl, edgecolors='none')
ax5.set_title('Vista lateral (X vs Y) — altura del rollo', fontsize=10)
ax5.set_xlabel('X'); ax5.set_ylabel('Y')
ax5.legend(fontsize=8, markerscale=1.4)

# ── 4f.  Varianza explicada por PCA ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
pca_full   = PCA().fit(X_scaled)
cum_var    = np.cumsum(pca_full.explained_variance_ratio_)
n_comp     = np.arange(1, len(cum_var) + 1)

ax6.bar(n_comp, pca_full.explained_variance_ratio_ * 100,
        color='#378ADD', alpha=0.7, label='Var. individual')
ax6.plot(n_comp, cum_var * 100,
         color='#E24B4A', lw=2, marker='o', ms=5, label='Var. acumulada')
ax6.axhline(90, color='gray', ls='--', lw=1, alpha=0.6)
ax6.text(2.1, 91.5, '90 %', color='gray', fontsize=8)
ax6.set_title('Varianza explicada por PCA\n(Swiss Roll 3D → 3 componentes)',
              fontsize=10)
ax6.set_xlabel('Número de componentes')
ax6.set_ylabel('Varianza explicada (%)')
ax6.set_xticks(n_comp)
ax6.legend(fontsize=9)
ax6.set_ylim(0, 105)

plt.savefig('swiss_roll_viz.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Figura guardada: swiss_roll_viz.png")

# ─────────────────────────────────────────────
# 5.  Figura secundaria: Moons en alta dimensión
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Moons en alta dimensión (R²→R³⁰) — Visualización',
              fontsize=14, fontweight='bold')

MOON_COLORS = ['#378ADD', '#E24B4A']
MOON_NAMES  = ['Luna 0', 'Luna 1']

# 5a.  Moons original 2D
for cls, col, lbl in zip(range(2), MOON_COLORS, MOON_NAMES):
    mask = y_moons == cls
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=col, s=12, alpha=0.7, label=lbl, edgecolors='none')
axes[0].set_title('Moons 2D original\n(estructura no lineal clara)', fontsize=10)
axes[0].set_xlabel('x₁'); axes[0].set_ylabel('x₂')
axes[0].legend(fontsize=9)

# 5b.  PCA sobre Moons HD (primeras 2 componentes)
pca_moons = PCA(n_components=2)
X_moons_pca = pca_moons.fit_transform(X_moons)
for cls, col, lbl in zip(range(2), MOON_COLORS, MOON_NAMES):
    mask = y_moons == cls
    axes[1].scatter(X_moons_pca[mask, 0], X_moons_pca[mask, 1],
                    c=col, s=12, alpha=0.7, label=lbl, edgecolors='none')
axes[1].set_title(
    f'PCA 2D sobre Moons HD (R³⁰)\n'
    f'PC1={pca_moons.explained_variance_ratio_[0]:.1%}  '
    f'PC2={pca_moons.explained_variance_ratio_[1]:.1%}',
    fontsize=10)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].legend(fontsize=9)

# 5c.  Varianza acumulada Moons HD
pca_moons_full = PCA().fit(X_moons)
cum_moons = np.cumsum(pca_moons_full.explained_variance_ratio_) * 100
axes[2].plot(range(1, len(cum_moons)+1), cum_moons,
             color='#1D9E75', lw=2, marker='o', ms=4)
for pct in [80, 90, 95]:
    idx = np.searchsorted(cum_moons, pct)
    axes[2].axhline(pct, color='gray', ls='--', lw=0.8, alpha=0.5)
    axes[2].text(1.2, pct + 0.8, f'{pct}%', color='gray', fontsize=8)
axes[2].set_title('Varianza explicada acumulada\n(Moons HD — 30 dims)', fontsize=10)
axes[2].set_xlabel('Número de componentes')
axes[2].set_ylabel('Varianza acumulada (%)')
axes[2].set_ylim(0, 105)
axes[2].grid(True, alpha=0.4)

fig2.tight_layout()
fig2.savefig('moons_hd_viz.png',
             dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Figura guardada: moons_hd_viz.png")

plt.show()
print("\nResumen:")
print(f"  Swiss Roll  — varianza PCA 2D: {sum(var_explained):.1%}")
print(f"  Moons HD    — varianza PCA 2D: {sum(pca_moons.explained_variance_ratio_):.1%}")
