import numpy as np

class Imputer:
    def __init__(self):
        '''
        1. 保存来了几批chunk或者来了多少数据
        2. 保存mean,mode
        '''
        self.k = None
        self.n_features = None
        self.count = None
        self.mean = None
        self.m2 = None
        
        pass
    
    def fit(self):
        pass
    
    def partial_fit(self,X_trunk):
        pass
    
    def transform(self, X_chunk, strategy):
        if strategy not in ( 'mean', 'mode'):
            raise ValueError('The strategy is wrong')
            
            
        if strategy == 'mean':
            # Turn the '' into np.nan
            X_chunk[X_chunk == ''] = np.nan
            
            means = np.nanmean(X_chunk, axis=0)
            idx = np.where(np.isnan(X_chunk))
            X_chunk[idx] = means[idx[1]]
            
            return X_chunk
        else:
            # Fill the missing value with the most frequent value
            for i in range(X_chunk.shape[1]):
                col = X_chunk[:,i]
                mask = (col == '')
                values, counts = np.unique(col[~mask], return_counts=True)
                value_mode = values[np.argmax(counts)]
                col[mask] = value_mode
            
            return X_chunk

class OneHotEncoder:
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def partial_fit(self,X_trunk):
        pass
    
    def transform(self, X_chunk):
        pass
    
class StandardScaler:
    def __init__(self):
        pass
    def fit(self):
        pass
    def partial_fit(self, X_chunk):
        pass
    def transform(self, X_chunk, strategy):
        pass
    
