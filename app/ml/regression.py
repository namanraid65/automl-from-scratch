import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000, fit_intercept=True, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.method = method
        self.weights = None
        self.bias = 0.0
        self.history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.history = []

        if self.method == 'ols':
            # Closed-form Ordinary Least Squares
            if self.fit_intercept:
                X_b = np.c_[np.ones((n_samples, 1)), X]
            else:
                X_b = X
            
            # Use pseudo-inverse for numerical stability
            try:
                theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            except np.linalg.LinAlgError:
                # Fallback if pinv fails
                theta = np.zeros((X_b.shape[1], 1))
                
            if self.fit_intercept:
                self.bias = float(theta[0, 0])
                self.weights = theta[1:].reshape(-1)
            else:
                self.bias = 0.0
                self.weights = theta.reshape(-1)
                
            # Compute final MSE
            y_pred = self.predict(X).reshape(-1, 1)
            final_loss = float(np.mean((y_pred - y) ** 2))
            self.history = [final_loss]
            
        else:
            # Gradient Descent
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            
            for epoch in range(self.epochs):
                y_pred = (X @ self.weights + self.bias).reshape(-1, 1)
                
                # Compute loss (MSE)
                loss = float(np.mean((y_pred - y) ** 2))
                self.history.append(loss)
                
                # Compute gradients
                dw = (2 / n_samples) * (X.T @ (y_pred - y)).reshape(-1)
                db = float((2 / n_samples) * np.sum(y_pred - y))
                
                # Clip gradients to avoid explosion
                dw = np.clip(dw, -1e3, 1e3)
                db = np.clip(db, -1e3, 1e3)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Early stopping if loss doesn't change much
                if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < 1e-9:
                    break
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


class MultipleLinearRegressionScratch(LinearRegressionScratch):
    """
    Multiple Linear Regression is mathematically identical to Linear Regression
    in terms of the matrix operations. We reuse the parent class.
    """
    pass


class PolynomialRegressionScratch:
    def __init__(self, degree=2, learning_rate=0.01, epochs=1000, fit_intercept=True, method='ols'):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.method = method
        self.base_regressor = LinearRegressionScratch(
            learning_rate=learning_rate, 
            epochs=epochs, 
            fit_intercept=fit_intercept, 
            method=method
        )
        self.history = []

    def _expand_features(self, X):
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape
        
        # Create polynomial features (power columns for each feature)
        # E.g., for [x1, x2] and degree 2, creates [x1, x2, x1^2, x2^2]
        # This keeps the feature space linear in size rather than exponential,
        # which is much more stable and memory-efficient for arbitrary tabular data.
        expanded_list = [X_arr]
        for d in range(2, self.degree + 1):
            expanded_list.append(X_arr ** d)
            
        return np.column_stack(expanded_list)

    def fit(self, X, y):
        X_poly = self._expand_features(X)
        self.base_regressor.fit(X_poly, y)
        self.history = self.base_regressor.history
        return self

    def predict(self, X):
        X_poly = self._expand_features(X)
        return self.base_regressor.predict(X_poly)
