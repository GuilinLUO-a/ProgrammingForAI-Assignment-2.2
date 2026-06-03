import numpy as np

class StreamingStats:
    def __init__(self):
        self.n_features = None
        self.count = None
        self.mean = None
        self.m2 = None

        
        # self.values = None
        
    def update_stats(self,X_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        if self.n_features is None:
            self.n_features = X_chunk.shape[1]
            self.count = np.zeros(self.n_features)
            self.mean = np.zeros(self.n_features)
            self.m2 = np.zeros(self.n_features)
            
            
        self.update_meanVar(X_chunk)
        # self.update_values(X_chunk)
    
    def update_meanVar(self,X_chunk):
        
        mask = ~np.isnan(X_chunk)
        if X_chunk[mask].size == 0:
            raise ValueError('The input data is empty')
        
        n_1 = self.count
        mean_1 = self.mean
        m2_1 = self.m2
        
        
        n_2 = mask.sum(axis=0)
        mean_2 = np.nanmean(X_chunk, axis=0)
        diff = np.where(mask, X_chunk - mean_2, 0.0)
        m2_2 = np.sum(diff **2, axis=0 )
        
        n_new = n_1 + n_2
        delta = mean_2 - mean_1
        n_safe = np.where(n_new > 0, n_new, 1)
        mean_new = mean_1 + delta * (n_2 / n_safe)
        m2_new = m2_1 + m2_2 + (delta ** 2) * (n_1 * n_2  / n_safe) 

        self.count = n_new
        self.mean = mean_new
        self.m2 = m2_new
        
    # def update_values(self,X_chunk):
    #     if self.values is None:
    #         self.values = X_chunk.T
    #     else:
    #         data = X_chunk.T
    #         self.values = np.concatenate(self.values, data, axis=1)
        
            
    
    def get_meanVar(self):
        if self.count is None:
            raise RuntimeError('The data has not been input yet')
        
        variance = self.m2 / self.count
        return self.mean, variance
    
    # def get_quantiles(self,quantiles):
    #     if quantiles < 0 or quantiles > 1:
    #         raise ValueError("The quantile value should be either greater than or equal to zero or less than or equal to one ")
        
    #     return np.quantile(self.values, quantiles, axis=1)
    
    # def get_histograms(self, bins=5):
    #     histograms = []
        
    #     for feature in self.values:
    #         hist, edges = np.histogram(feature, bins)
    #         histograms.append((hist, edges))
            
    #     return histograms
    
    
          
        