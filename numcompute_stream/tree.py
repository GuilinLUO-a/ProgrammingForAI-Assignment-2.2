class DecisionTreeClassifier:

    def __init__(self,
                 max_depth,
                 min_samples_split,
                 max_features,
                 criterion="gini"):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
    
    def partial_fit(X_chunk, y_chunk):
        ...
    
    