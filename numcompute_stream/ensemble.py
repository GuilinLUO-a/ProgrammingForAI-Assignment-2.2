import numpy as np
from collections import Counter

class EnsembleClassifier:

    def __init__(self, base_tree, n_estimators=5,  **tree_args):
        self.n_estimators = n_estimators

        self.trees = [
            base_tree(**tree_args)
            for _ in range(self.n_estimators)]
        
    def partial_fit(self, X_chunk, y_chunk):
        n_samples = X_chunk.shape[0]
        
        for tree in self.trees:
            idx = np.random.choice(n_samples, n_samples, replace=True)

            X_replacement = X_chunk[idx]
            y_replacement = y_chunk[idx]
            
            tree.partial_fit(X_replacement, y_replacement)

    def predict(self, X):
        predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])
        
        prediction_majority = np.array([
            Counter(predictions[:,i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])
        
        return prediction_majority
        
    