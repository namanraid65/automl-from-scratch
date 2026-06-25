import numpy as np

class KNNClassifierScratch:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.history = [0] # Dummy training history for UI progress consistencies

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y).reshape(-1)
        self.history = [0.0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict_proba(self, X):
        # Return probability matrix
        X = np.asarray(X, dtype=float)
        unique_classes = np.unique(self.y_train)
        probs = []
        for x in X:
            if self.metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            p_dict = {c: 0.0 for c in unique_classes}
            for label in k_nearest_labels:
                if label in p_dict:
                    p_dict[label] += 1.0 / self.k
            probs.append([p_dict[c] for c in unique_classes])
        return np.array(probs)


class BinaryLogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.history = []

    def _sigmoid(self, z):
        # Clip z to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []

        for epoch in range(self.epochs):
            z = (X @ self.weights + self.bias).reshape(-1, 1)
            y_pred = self._sigmoid(z)
            
            # Compute Binary Cross Entropy Loss
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -float(np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))
            self.history.append(loss)
            
            # Compute Gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y)).reshape(-1)
            db = float((1 / n_samples) * np.sum(y_pred - y))
            
            # Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping
            if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < 1e-9:
                break
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)


class LogisticRegressionScratch:
    """
    Multi-Class Logistic Regression using One-vs-Rest (OvR) scheme.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classifiers = {}
        self.classes_ = None
        self.history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.classifiers = {}
        self.history = []

        # If it's a binary classification problem, we can train a single binary classifier
        # However, for simplicity and uniformity of API, we can train binary classifiers for each class vs all.
        # Wait, if there are exactly 2 classes, say 0 and 1, we can just train one binary classifier.
        # Let's write binary logic if len(classes) == 2, else OvR.
        # Actually, standard OvR works for binary as well, but lets optimize and support multi-class directly.
        
        # We will collect the histories of each binary classifier and average them for the final history.
        all_histories = []
        
        if len(self.classes_) == 2:
            # Binary Case
            c0, c1 = self.classes_[0], self.classes_[1]
            y_binary = (y == c1).astype(int)
            clf = BinaryLogisticRegressionScratch(learning_rate=self.learning_rate, epochs=self.epochs)
            clf.fit(X, y_binary)
            self.classifiers['binary'] = clf
            self.history = clf.history
        else:
            # Multi-class OvR Case
            for c in self.classes_:
                y_binary = (y == c).astype(int)
                clf = BinaryLogisticRegressionScratch(learning_rate=self.learning_rate, epochs=self.epochs)
                clf.fit(X, y_binary)
                self.classifiers[c] = clf
                all_histories.append(clf.history)
                
            # Align histories (since they can stop early at different lengths)
            max_len = max(len(h) for h in all_histories)
            padded_histories = []
            for h in all_histories:
                pad_width = max_len - len(h)
                if pad_width > 0:
                    padded_histories.append(h + [h[-1]] * pad_width)
                else:
                    padded_histories.append(h)
            self.history = list(np.mean(padded_histories, axis=0))
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if len(self.classes_) == 2:
            prob_class_1 = self.classifiers['binary'].predict_proba(X)
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            probs = []
            for c in self.classes_:
                probs.append(self.classifiers[c].predict_proba(X))
            probs = np.column_stack(probs)
            # Normalize to make sure they sum to 1
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-8
            return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]


class NaiveBayesScratch:
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        self.history = [0.0]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        self.class_priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        self.history = [0.0]

        for c in self.classes_:
            X_c = X[y == c]
            self.class_priors_[c] = len(X_c) / n_samples
            
            # Compute means and variances for each feature
            # Add a small epsilon to variance to avoid division by zero
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + 1e-9
            
        return self

    def _calculate_likelihood(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        
        # Gaussian PDF: (1 / sqrt(2*pi*var)) * exp(-(x-mean)^2 / (2*var))
        # Log version for numerical stability:
        # -0.5 * log(2*pi*var) - (x-mean)^2 / (2*var)
        numerator = -((x - mean) ** 2) / (2 * var)
        denominator = -0.5 * np.log(2 * np.pi * var)
        return numerator + denominator

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes_:
            prior = np.log(self.class_priors_[c])
            # Sum log-likelihoods of all features
            likelihood = np.sum(self._calculate_likelihood(c, x))
            posteriors.append(prior + likelihood)
        return self.classes_[np.argmax(posteriors)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = []
        for x in X:
            posteriors = []
            for c in self.classes_:
                prior = np.log(self.class_priors_[c])
                likelihood = np.sum(self._calculate_likelihood(c, x))
                posteriors.append(prior + likelihood)
            
            # Convert log probabilities back to normal probabilities safely using log-sum-exp trick
            posteriors = np.array(posteriors)
            max_post = np.max(posteriors)
            exp_post = np.exp(posteriors - max_post)
            prob = exp_post / np.sum(exp_post)
            probs.append(prob)
        return np.array(probs)
