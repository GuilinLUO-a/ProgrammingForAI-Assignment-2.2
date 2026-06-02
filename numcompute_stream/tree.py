import numpy as np
from collections import deque, Counter

class DecisionTreeClassifier:

    def __init__(self, max_depth, min_samples_split, max_features, criterion="Entropy", window_size = None):

        if criterion not in ('Entropy','Gini'):
            raise ValueError('Criterion should be either Entropy or Gini')
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
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
            return Counter(y).most_common(1)[0][0]  # return most common label 

        feat, thresh = self._best_split(X, y)             # find best split

        if feat is None:                            # no valid split is found
            return Counter(y).most_common(1)[0][0]  # fallback leaf

        left_idx = X[:, feat] <= thresh             # left split indices
        right_idx = X[:, feat] > thresh             # right split indices

        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)   # build left subtree
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1) # build right subtree

        return (feat, thresh, left, right)
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])  # apply to all samples
    
    def _predict_one(self, x, node=None):            # node represents the current position in the tree
        if node is None:
            node = self.tree                        # start from root

        if not isinstance(node, tuple):
            return node                             # return label if leaf

        feat, thresh, left, right = node           # unpack node

        if x[feat] <= thresh:
            return self._predict_one(x, left)       # go left
        else:
            return self._predict_one(x, right)      # go right
        
    def _entropy(self,y):
        _, counts = np.unique(y, return_counts=True)  # count how many times each class appears
        probs = counts / counts.sum()                 # convert counts into probabilities
        return -np.sum(probs * np.log2(probs + 1e-9)) # apply entropy formula (add small value to avoid log(0))
    
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
        best_gain = -1                # keep track of best gain found so far
        best_feat = None             # best feature index
        best_thresh = None           # best threshold value

        features = np.arange(X.shape[1])
        
        if self.max_features is not None:
            features = np.random.choice(features, size=self.max_features, replace=False)
        
        for feat in features:                     # loop over all features
            thresholds = np.unique(X[:, feat])             # get possible split values
            for thresh in thresholds:                      # try each threshold

                left = y[X[:, feat] <= thresh]             # labels for left split
                right = y[X[:, feat] > thresh]             # labels for right split

                if len(left) == 0 or len(right) == 0:      # skip invalid splits
                    continue
                
                
                gain = self._impurity_gain(y, left, right)

                if gain > best_gain:                       # update best split if better
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh
    