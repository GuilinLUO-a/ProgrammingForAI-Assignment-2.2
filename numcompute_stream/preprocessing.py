import numpy as np
from stats import StreamingStats

class Imputer:
    def __init__(self, strategy):
        if strategy not in ( 'mean', 'mode'):
            raise ValueError('The strategy is wrong')
        
        self.strategy = strategy
        self.means = None
        self.categories_counts = None
        self.fill_values = None
        self.stats = StreamingStats()
    
    def partial_fit(self,X_chunk):
        if X_chunk.size == 0:
            raise ValueError('The input data is empty')
        
        if self.strategy == 'mean':
            self.stats.update_stats(X_chunk)
            means, _ = self.stats.get_meanVar()
            self.means = means
            
        else:
            if self.categories_counts is None:
                self.categories_counts = [{} for i in range(X_chunk.shape[1])]
            
            for i in range(X_chunk.shape[1]):
                col = X_chunk[:,i]
                mask = (col == '')
                
                if len(mask) > 0:
                    values, counts = np.unique(col[~mask], return_counts=True)
                    for v,c in zip(values, counts):
                        self.categories_counts[i][v] = self.categories_counts[i].get(v, 0) + int(c)

            # This will get the first largest number of classification it met when facing tie situation
            self.fill_values = [
                max(values, key=values.get)
                if values else None
                for values in self.categories_counts
            ]
        
        return self
    
    def fit(self,X_chunk):
        return self.partial_fit(X_chunk)
            
    def transform(self, X_chunk):
        if self.means is None :
            if self.fill_values is None:
                raise RuntimeError('You should fit your data first')
        
        X_chunk = X_chunk.copy()
            
        if self.strategy == 'mean':
            X_chunk = np.where(~np.isnan(X_chunk), X_chunk, self.means)
            
            return X_chunk
        elif self.strategy == 'mode':
            # Fill the missing value with the most frequent value
            for i in range(X_chunk.shape[1]):
                col = X_chunk[:,i]
                mask = (col == '')
                col[mask] = self.fill_values[i]
            
            return X_chunk

class OneHotEncoder:
    def __init__(self):
        self.categories = None
        self.n_features = None
        self.locked = False
        
    
    def fit(self,X_chunk):
        return self.partial_fit(X_chunk)
    
    def partial_fit(self,X_chunk):
        if X_chunk.size == 0:
            raise ValueError('The input data is empty')
        if X_chunk.ndim!=2:
            raise ValueError('The dimension should be 2D')
        if self.n_features is not None and X_chunk.shape[1] != self.n_features:
            raise ValueError(f'The features should be {self.n_features}')
            
        if not self.locked:
            self.categories = [
                np.unique(X_chunk[:,i])
                for i in range(X_chunk.shape[1])
            ]
            self.n_features = X_chunk.shape[1]
            self.locked = True
            
        return self
    
    def transform(self, X_chunk):
        if self.n_features is None:
            raise  RuntimeError('Should call partial_fit() first')

        result = []
        
        for i in range(X_chunk.shape[1]):
            categories = self.categories[i]
            col = X_chunk[:,i]
            
            n_categories = len(categories)
            cat_idx = {cat:idx for idx,cat in enumerate(categories)}

            indices = np.array([
                cat_idx.get(val, n_categories)
                for val in col
            ])
            
            identity = np.eye(n_categories + 1) # Used for unknown category
            result.append(identity[indices])
            
        return np.concatenate(result, axis=1)
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_features = None
        self.stats = StreamingStats()
        
    def fit(self,X_chunk):
        return self.partial_fit(X_chunk)

    def partial_fit(self, X_chunk):
        X_chunk = X_chunk.copy()
        
        if X_chunk.size == 0:
            raise ValueError('The input data is empty')
        if X_chunk.ndim != 2:
            raise ValueError('The dimension should be 2D')
        if self.n_features is not None and X_chunk.shape[1] != self.n_features:
            raise ValueError(f'The features should be {self.n_features}')
        

        if self.n_features is None:
            self.n_features = X_chunk.shape[1]
        
        self.mean, variance = self.stats.get_meanVar()
        std = np.sqrt(variance)
        self.std = std
        
        return self
        
    def transform(self, X_chunk):
        X_chunk = X_chunk.copy()
        
        if self.n_features is None:
            raise  RuntimeError('Should call partial_fit() first')
        
        if X_chunk.size == 0:
            raise ValueError('The input data is empty')
        if X_chunk.ndim != 2:
            raise ValueError('The dimension should be 2D')
        if self.n_features is not None and X_chunk.shape[1] != self.n_features:
            raise ValueError(f'The features should be {self.n_features}')
        
        X_chunk = (X_chunk - self.mean) / self.std
        
        return X_chunk
    
