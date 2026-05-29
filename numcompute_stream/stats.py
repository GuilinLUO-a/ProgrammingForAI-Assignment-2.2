import numpy as np

class StreamingStats:
    def __init__(self):
        self.n_features = None
        self.count = None
        self.mean = None
        self.m2 = None
        
        self.values = None
        
    def update_stats(self,X_chunk):
        X_chunk = np.asarray(dtype=float)
        if self.n_features is None:
            self.n_features = X_chunk.shape[1]
            self.count = np.zeros(self.n_features)
            self.mean = np.zeros(self.n_features)
            self.m2 = np.zeros(self.n_features)
            
        self.update_meanVar(X_chunk)
        self.update_values(X_chunk)
    
    def update_meanVar(self,X_chunk):
        
        n_b = X_chunk.shape[0]
        mean_b = np.mean(X_chunk, axis=0)
        m2_b = np.sum((X_chunk - mean_b)**2, axis=0 )
        
        n = self.count + n_b
        mean = self.mean + (n_b / n) * (mean_b - self.mean) 
        m2 = self.m2 + m2_b + (((mean_b - self.mean) **2) * (self.count*n_b)/n)  
        
        self.count = n
        self.mean = mean
        self.m2 = m2 
    
    def update_values(self,X_chunk):
        if self.values is None:
            self.values = X_chunk.T
        else:
            data = X_chunk.T
            self.values = np.concatenate(self.values, data, axis=1)
        
            
    
    def get_meanVar(self):
        variance = self.m2 / self.count
        return self.mean, variance
    
    def get_quantiles(self,quantiles):
        if quantiles < 0 or quantiles > 1:
            raise ValueError("The quantile value should be either greater than or equal to zero or less than or equal to one ")
        
        return np.quantile(self.values, quantiles, axis=1)
    
    def get_histograms(self, bins=5):
        histograms = []
        
        for feature in self.values:
            hist, edges = np.histogram(feature, bins)
            histograms.append((hist, edges))
            
        return histograms
    
    
          
        