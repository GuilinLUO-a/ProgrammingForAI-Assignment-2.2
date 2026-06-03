import numpy as np
from preprocessing import Imputer, StandardScaler, OneHotEncoder

class PipeLine:
    def __init__(self, steps):
        self.steps = steps
        self.label_encoder = LabelEncoder()

    def partial_fit(self, X, y):
        self.label_encoder.partial_fit(X, y)
        y_encoded = self.label_encoder.transform(y)

        transformers = self.steps[:-1]
        model = self.steps[-1][1]
        
        for _, step in transformers:
            step.partial_fit(X)
            X = step.transform(X)

        model.partial_fit(X, y_encoded)
        return self
        
    
    def fit(self, X, y):
        return self.partial_fit(X, y)
    
    def predict(self, X):
        transformers = self.steps[:-1]
        model = self.steps[-1][1]
        
        for _, step in transformers:
            X = step.transform(X)

        y_encoded = model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
        
    def score(self, X_pred, y_true):
        y_pred = self.predict(X_pred)
        return np.mean(y_pred == np.asarray(y_true))
    
    
class Preprocessor:
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None
        self.num_imputer = Imputer(strategy='mean')
        self.cat_imputer = Imputer(strategy='mode')
        self.scaler = StandardScaler()
        self.enoder = OneHotEncoder()

    def _infer_column(self, X):
        num_cols = []
        cat_cols = []

        for i in range(X.shape[1]):
            col = X[:,i]
            
            try:
                col[col != ''].astype(float)
                num_cols.append(i)
            except:
                cat_cols.append(i)

        return num_cols, cat_cols
    
    def partial_fit(self, X):
        if X.shape[0] == 0:
            raise ValueError('You input empty dataset')

        self.num_cols, self.cat_cols = self._infer_column(X) 

        if self.num_cols:
            X_num = X[:, self.num_cols].astype(float)
            self.num_imputer.partial_fit(X_num)
            self.scaler.partial_fit(self.num_imputer.transform(X_num))

        if self.cat_cols:
            X_cat = X[:,self.cat_cols]
            self.cat_imputer.partial_fit(X_cat)
            self.enoder.partial_fit(self.cat_imputer.transform(X_cat))
            
        return self
    
    def fit(self, X):
        return self.partial_fit(X)
    
    def transform(self, X):
        if self.num_cols is None:
            raise RuntimeError('You should call partial_fit() or fit() function first')

        X_transformed = []    

        if self.num_cols:
            X_num = X[:,self.num_cols].astype(float)
            X_num = self.num_imputer.transform(X_num)
            X_num = self.scaler.transform(X_num)

            X_transformed.append(X_num)
        
        if self.cat_cols:
            X_cat = X[:,self.cat_cols]
            X_cat = self.cat_imputer.transform(X_cat)
            X_cat = self.enoder.transform(X_cat)

            X_transformed.append(X_cat)
        
        return np.concatenate(X_transformed, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reset(self):
        self.num_cols = None
        self.cat_cols = None
        self.num_imputer = Imputer(strategy='mean')   
        self.cat_imputer = Imputer(strategy='mode')   
        self.scaler = StandardScaler()                
        self.enoder = OneHotEncoder() 
    
class LabelEncoder:
    def __init__(self):
        self.class_to_idx = {}
        self.idx_to_class = {}

    def partial_fit(self, y):
        y = np.asarray(y)
        
        for c in y:
            if c not in self.class_to_idx:
                idx = len(self.class_to_idx)
                self.class_to_idx[c] = idx
                self.idx_to_class[idx] = c
                
    def fit(self, y):
        return self.partial_fit(y)
    
    def transform(self, y):
        return np.array([self.class_to_idx[c] for c in y])
    
    def inverse_transform(self, y):
        return np.array([self.idx_to_class[idx] for idx in y])