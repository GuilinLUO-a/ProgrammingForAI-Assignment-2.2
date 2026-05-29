class EnsembleClassifier:

    def __init__(self, 
                 n_trees, 
                 max_depth,
                 min_samples_split,
                 max_features,
                 method,
                 criterion="gini"):
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.method = method
        self.criterion = criterion
        

    def partial_fit(self, X_chunk, y_chunk):
        ...

    def predict(self, X):
        ...