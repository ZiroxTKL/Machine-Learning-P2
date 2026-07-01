"""
Support Vector Machine
--------------------
Implementation of an example that show how to train a SVM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

print("\n\nBasic implementation about Support Vector Machine (SVM)\n\n")

# Create dataset
#np.random.seed(42)   # What happen if I quit the comment here question 1?
N = 50
X_pos = np.random.randn(N,2) + np.array([2,2])
X_neg = np.random.randn(N,2) + np.array([-2,-2])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(N), -np.ones(N)])
X += np.random.normal(0, 0.1, X.shape)

# Split for training/testing
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# What happen if I change the comment here question 2?

# Cretae a class Linear SVM that develop the theory in our slides
class LinearSVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.y[idx] * linear_model >= 1

    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db

        dw = self.lambda_param * self.w - np.dot(self.y[idx], x)
        db = - self.y[idx]
        return dw, db

    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.y = self._get_cls_map(y)   

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                constrain = self._satisfy_constraint(x_i, idx)
                if constrain:
                    self.w -= self.lr * (self.lambda_param * self.w)
                    self.b -= 0
                else:
                    self.w -= self.lr * (self.lambda_param * self.w - np.dot(x_i, self.y[idx]))
                    self.b -= self.lr * -y[idx]

    def get_support_vectors(self, X, y):
        support_vector_idx = np.where(np.abs(y * (np.dot(X, self.w) + self.b) - 1) <= 0.05)[0]
        return support_vector_idx

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)
        #return prediction
        # What happen if I change the comment here question 3?

# Test my class SVM  
model_svm = LinearSVM(lambda_param=1e-5)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Plotting my test cases
import numpy as np
import matplotlib.pyplot as plt

# Plot SVM training test
def plot_svm(X_train, y_train, X_test, y_test, model, title="SVM"):
    support_vector_idx = model.get_support_vectors(X_train, y_train)
    w, b = model.w, model.b

    plt.figure(num=title, figsize=(8,6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=['red' if yi==1 else 'yellow' for yi in y_train], s=50)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=['darkred' if yi==1 else 'orange' for yi in y_test], s=50, marker='*')
    plt.scatter(X_train[support_vector_idx][:, 0], X_train[support_vector_idx][:, 1],
                s=150, linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    ax = plt.gca()
    xlim = ax.get_xlim()
    xx = np.linspace(*xlim)
    yy = -(w[0] * xx + b) / w[1]
    margin = 1 / np.linalg.norm(w)
    offset = np.sqrt(1 + (w[0]/w[1])**2) * margin
    yy_down = yy - offset
    yy_up = yy + offset

    plt.plot(xx, yy, 'k-', label='Decision boundary')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.legend()
    plt.title("Linearly Separable Soft Margin SVM (Primal Form)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show(block=False)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(num=title, figsize=(5,5))
    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    classes = [-1, 1]

    plt.xticks([0,1], classes)
    plt.yticks([0,1], classes)

    # Mostrar valores dentro de la matriz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]),
                     ha='center',
                     va='center',
                     fontsize=14)

    plt.colorbar()
    plt.show(block=False)
    

# lambda_param=10
model_svm = LinearSVM(lambda_param=10)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

plot_confusion_matrix(
    y_test,
    y_pred,
    title="Confusion Matrix - lambda=10"
)

plot_svm(X_train, y_train, X_test, y_test, model_svm, title="SVM - lambda=10")

# lambda_param=1
model_svm = LinearSVM(lambda_param=1)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

plot_confusion_matrix(
    y_test,
    y_pred,
    title="Confusion Matrix - lambda=1"
)

plot_svm(X_train, y_train, X_test, y_test, model_svm, title="SVM - lambda=1")

# lambda_param=1e-2
model_svm = LinearSVM(lambda_param=1e-2)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

plot_confusion_matrix(
    y_test,
    y_pred,
    title="Confusion Matrix - lambda=1e-2"
)

plot_svm(X_train, y_train, X_test, y_test, model_svm, title="SVM - lambda=1e-2")

