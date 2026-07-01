"""
Support Vector Machine - Soft Margin
-------------------------------------
Implementation of a Soft Margin SVM using slack variables ξᵢ
based on the formulation:
    min  (1/2)||w||² + λ * Σ ξᵢ
    s.t  ξᵢ >= 0;  ∀i  t⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) >= 1 - ξᵢ
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
 
print("\n\nSoft Margin SVM with Slack Variables ξᵢ\n\n")
 
# ── Dataset ────────────────────────────────────────────────────────────────────
X = np.array([
    # Class +1 — correctly classified (6 points, well beyond margin)
    [2.5,  3.1],
    [3.2,  1.4],
    [4.0,  2.8],
    [3.7, -0.5],
    [2.8, -2.1],
    [4.5,  0.2],
    # Class +1 — inside margin (2 points)
    [0.4,  1.2],
    [0.7, -1.8],
    # Class +1 — wrong side (1 point)
    [-0.6,  0.5],
    # Class -1 — correctly classified (6 points)
    [-2.5,  2.9],
    [-3.1,  0.7],
    [-4.0, -1.2],
    [-3.5,  2.0],
    [-2.8, -0.8],
    [-4.2,  1.5],
    # Class -1 — inside margin (2 points)
    [-0.5,  2.3],
    [-0.8, -1.5],
    # Class -1 — wrong side (1 point)
    [0.5, -0.9],
])
 
y = np.array([1, 1, 1, 1, 1, 1,  1, 1,  1,
             -1,-1,-1,-1,-1,-1, -1,-1, -1])
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 
# ── Linear SVM class (Soft Margin) ────────────────────────────────────────────
class LinearSVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr           = learning_rate
        self.lambda_param = lambda_param
        self.n_iters      = n_iters
        self.w            = None
        self.b            = None
 
    # Map labels to {-1, +1}
    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)
 
    # ── SOFT MARGIN: compute slack variable ξᵢ for one sample ──────────────
    # From the slide:  ξᵢ = max(0, 1 - yᵢ·f(xᵢ))
    #   ξᵢ = 0        → point is outside the margin, correctly classified
    #   0 < ξᵢ <= 1   → point is inside the margin (margin violation)
    #   ξᵢ > 1        → point is on the wrong side of the hyperplane
    def _compute_slack(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        xi = max(0, 1 - self.y[idx] * linear_model)
        return xi
 
    # ── Gradient rules derived from the soft-margin loss ───────────────────
    # Loss = (1/2)||w||² + λ * Σ ξᵢ
    # If ξᵢ = 0  (constraint satisfied):
    #     ∂Loss/∂w = λ·w       ∂Loss/∂b = 0
    # If ξᵢ > 0  (constraint violated):
    #     ∂Loss/∂w = λ·w - yᵢ·xᵢ    ∂Loss/∂b = -yᵢ
    def _get_gradients(self, xi, x, idx):
        if xi == 0:
            dw = self.lambda_param * self.w
            db = 0
        else:
            dw = self.lambda_param * self.w - np.dot(self.y[idx], x)
            db = -self.y[idx]
        return dw, db
 
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
 
    # ── Training loop ───────────────────────────────────────────────────────
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.y = self._get_cls_map(y)      # store mapped labels
 
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                xi = self._compute_slack(x_i, idx)   # ← compute ξᵢ
                dw, db = self._get_gradients(xi, x_i, idx)
                self._update_weights_bias(dw, db)
 
    # ── Support vectors: points where |yᵢ·f(xᵢ) - 1| ≈ 0  (on the margin) ─
    def get_support_vectors(self, X, y):
        support_vector_idx = np.where(
            np.abs(y * (np.dot(X, self.w) + self.b) - 1) <= 0.05
        )[0]
        return support_vector_idx
 
    # ── Return ξᵢ for every training point ─────────────────────────────────
    def get_slack_variables(self, X, y):
        y_mapped = self._get_cls_map(y)
        slacks = np.zeros(len(X))
        for i, x_i in enumerate(X):
            slacks[i] = max(0, 1 - y_mapped[i] * (np.dot(x_i, self.w) + self.b))
        return slacks
 
    # ── Prediction: sign(w·x + b) ──────────────────────────────────────────
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        return np.sign(estimate)
 
 
# ── Plotting functions ─────────────────────────────────────────────────────────
 
def plot_svm(X_train, y_train, X_test, y_test, model, title="SVM"):
    support_vector_idx = model.get_support_vectors(X_train, y_train)
    slacks             = model.get_slack_variables(X_train, y_train)
    w, b               = model.w, model.b
 
    plt.figure(num=title, figsize=(9, 6))
 
    # Training points
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=['red' if yi == 1 else 'yellow' for yi in y_train],
                s=50, zorder=3)
 
    # Test points
    plt.scatter(X_test[:, 0], X_test[:, 1],
                c=['darkred' if yi == 1 else 'orange' for yi in y_test],
                s=50, marker='*', zorder=3)
 
    # Support vectors — colored by class
    sv_colors = ['red' if y_train[i] == 1 else 'yellow' for i in support_vector_idx]
    plt.scatter(X_train[support_vector_idx][:, 0],
                X_train[support_vector_idx][:, 1],
                s=150, linewidth=1.5, facecolors='none',
                edgecolors=sv_colors, label='Support Vectors', zorder=4)
 
    # Hyperplane and margin lines
    ax   = plt.gca()
    xlim = ax.get_xlim()
    xx   = np.linspace(*xlim)
    yy   = -(w[0] * xx + b) / w[1]
 
    margin = 1 / np.linalg.norm(w)
    offset = np.sqrt(1 + (w[0] / w[1]) ** 2) * margin
 
    yy_down = yy - offset
    yy_up   = yy + offset
 
    plt.plot(xx, yy,      color='black', linestyle='-',  linewidth=2,   label='Decision boundary')
    plt.plot(xx, yy_down, color='red',   linestyle='--', linewidth=1.5, label='Margin −1')
    plt.plot(xx, yy_up,   color='blue',  linestyle='--', linewidth=1.5, label='Margin +1')
 
    # ── Draw ξᵢ segments for points that violate the margin ─────────────────
    # For each point with ξᵢ > 0, draw a dotted line from the point to
    # its closest margin boundary, and annotate with the ξᵢ value.
    for i, (x_i, xi) in enumerate(zip(X_train, slacks)):
        if xi > 0:
            # y-coordinate of the margin line at x_i[0]
            margin_boundary_y = -(w[0] * x_i[0] + b) / w[1]
            if y_train[i] == 1:
                margin_boundary_y += offset    # margin +1
            else:
                margin_boundary_y -= offset    # margin -1
 
            plt.plot([x_i[0], x_i[0]], [x_i[1], margin_boundary_y],
                     color='gray', linewidth=1.2, linestyle=':', alpha=0.8)
            plt.text(x_i[0] + 0.08,
                     (x_i[1] + margin_boundary_y) / 2,
                     f'ξ={xi:.2f}', fontsize=7, color='dimgray')
 
    total_slack = slacks.sum()
    plt.title(f"{title}  |  ∑ξᵢ = {total_slack:.3f}  (λ={model.lambda_param})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
 
 
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
 
    plt.figure(num=title, figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
 
    classes = [-1, 1]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=14)
 
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)
 
 
# ── Experiments with different λ values ───────────────────────────────────────
for lam in [10, 1, 1e-2, 1e-5]:
    model_svm = LinearSVM(lambda_param=lam)
    model_svm.fit(X_train, y_train)
 
    y_pred   = model_svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
 
    slacks      = model_svm.get_slack_variables(X_train, y_train)
    total_slack = slacks.sum()
    n_violated  = np.sum(slacks > 0)
 
    print(f"λ={lam:<8}  Accuracy={accuracy:.2f}  "
          f"∑ξᵢ={total_slack:.3f}  Points with ξᵢ>0: {n_violated}")
 
    label = str(lam)
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - lambda={label}")
    plot_svm(X_train, y_train, X_test, y_test, model_svm, title=f"SVM - lambda={label}")
 
plt.show()
