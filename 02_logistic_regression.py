"""Logistic Regression Implementation"""

import numpy as np


class LogisticRegression:
    """Logistic Regression using Gradient Descent for binary classification."""

    def __init__(self, learning_rate=0.1, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid: avoids overflow for large |z|
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.n_iterations):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            # Binary cross-entropy loss (clipped for log stability)
            eps = 1e-15
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.losses.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    # Synthetic 2D dataset: two Gaussian clusters
    np.random.seed(0)
    X = np.vstack([
        np.random.randn(20, 2) + np.array([2, 2]),
        np.random.randn(20, 2) + np.array([-2, -2]),
    ])
    y = np.hstack([np.zeros(20), np.ones(20)])

    model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
    model.fit(X, y)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Final loss:", round(model.losses[-1], 4))
    print("Accuracy:", round(model.score(X, y), 4))
