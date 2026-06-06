import numpy as np
from collections import deque, Counter

class DecisionTreeClassifier:

    def __init__(self, max_depth, min_samples_split, max_features=None, n_class=2, criterion="Entropy", window_size = None):

        if criterion not in ('Entropy','Gini'):
            raise ValueError('Criterion should be either Entropy or Gini')
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_class = n_class
        self.criterion = criterion
        self.tree = None

        self.window_size = window_size
        if window_size is not None:
            self.X_buffer = deque(maxlen=self.window_size) 
            self.y_buffer = deque(maxlen=self.window_size)
    
    def partial_fit(self, X_chunk, y_chunk):
        if self.window_size is not None:
            self.X_buffer.append(X_chunk)
            self.y_buffer.append(y_chunk)

            X_chunk = np.concatenate(self.X_buffer, axis=0)
            y_chunk = np.concatenate(self.y_buffer, axis=0)
        
        self.tree = self._grow_tree(X_chunk, y_chunk, depth=0)

    def _grow_tree(self, X, y, depth):
        if (depth >= self.max_depth 
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]   

        feat, thresh = self._best_split(X, y)             

        if feat is None:                            
            return Counter(y).most_common(1)[0][0]  

        left_idx = X[:, feat] <= thresh             
        right_idx = X[:, feat] > thresh             

        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)   
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1) 

        return (feat, thresh, left, right)
    
    def predict(self, X):
        if self.tree is None:
            return np.zeros(X.shape[0])
        
        return np.array([self._predict_one(x) for x in X])  
    
    def _predict_one(self, x, node=None):            
        if node is None:
            node = self.tree                        

        if not isinstance(node, tuple):
            return node                             

        feat, thresh, left, right = node           

        if x[feat] <= thresh:
            return self._predict_one(x, left)       
        else:
            return self._predict_one(x, right)      
        
    def _entropy(self,y):
        _, counts = np.unique(y, return_counts=True)  
        probs = counts / counts.sum()                 
        return -np.sum(probs * np.log2(probs + 1e-9)) 
    
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _impurity_gain(self, y, y_left, y_right):
        if self.criterion == 'Entropy':
            return self._entropy(y) - (len(y_left)/len(y))*self._entropy(y_left) - (len(y_right)/len(y))*self._entropy(y_right)
        elif self.criterion == 'Gini':
            return self._gini(y) - (len(y_left) / len(y) * self._gini(y_left)) - (len(y_right) / len(y) * self._gini(y_right))
        
    def _best_split(self, X, y):
        best_gain = -1                
        best_feat = None             
        best_thresh = None           

        features = np.arange(X.shape[1])
        
        if self.max_features is not None:
            features = np.random.choice(features, size=self.max_features, replace=False)
        
        for feat in features:                    
            thresholds = np.unique(X[:, feat])             
            for thresh in thresholds:                      

                left = y[X[:, feat] <= thresh]             
                right = y[X[:, feat] > thresh]            

                if len(left) == 0 or len(right) == 0:     
                    continue
                
                
                gain = self._impurity_gain(y, left, right)

                if gain > best_gain:                       
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh
    