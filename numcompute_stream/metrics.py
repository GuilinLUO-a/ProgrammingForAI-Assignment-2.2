import numpy as np
from collections import deque

class StreamMetrics:
    def __init__(self, n_class = 2, window_size=None):
        self.n_class = n_class
        self.window_size = window_size
        self.cm = None # Confusion matrix
        self.n_correct = None
        self.n_samples = None
        self.y_true = []
        self.y_score = []
        
        self.rolling_cm = None
        self.rolling_true = None
        self.rolling_score = None

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        if self.cm is None:
            self.reset()
            
        y_true_chunk = np.asarray(y_true_chunk)
        y_pred_chunk = np.asarray(y_pred_chunk)
        
        if y_true_chunk.shape != y_pred_chunk.shape:
            raise ValueError("The shape of two arrays are not the same")    

        cm = np.bincount(self.n_class * y_true_chunk + y_pred_chunk, 
                         minlength= self.n_class * self.n_class).reshape(self.n_class, self.n_class)

        self.cm += cm

        n_correct = np.diag(cm).sum() # The number of correct samples
        n_samples = cm.sum()          # The number of all samples

        self.n_correct += n_correct
        self.n_samples += n_samples
        
        if self.window_size is not None:
            self.rolling_cm.append(cm)
            
        if y_score_chunk is not None:
            y_score_chunk = np.asarray(y_score_chunk, dtype=float)

            for yt, ys in zip(y_true_chunk, y_score_chunk):
                self.y_true.append(yt)
                self.y_score.append(ys)

                if self.window_size is not None:
                    self.rolling_true.append(yt)
                    self.rolling_score.append(ys)

        return self

    def reset(self):
        self.cm = np.zeros((self.n_class, self.n_class), dtype=np.int64)
        self.n_correct = 0
        self.n_samples = 0
        self.y_true = []
        self.y_score = []
        
        if self.window_size is not None:
            self.rolling_cm = deque(maxlen=self.window_size)
            self.rolling_true = deque(maxlen=self.window_size)
            self.rolling_score = deque(maxlen=self.window_size)

    def accuracy(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        cm = self.cm
        
        return np.where(cm.sum()>0,np.diag(cm).sum() / cm.sum(),0.0)
        
    def precision(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        cm = self.cm
        
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        
        return np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        
    def recall(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        cm = self.cm
        
        TP = np.diag(cm)
        FN = cm.sum(axis=1) - TP
        
        return np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
    
    def f1(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        precision = self.precision()
        recall = self.recall()

        f1 = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)
        
        return f1
    
    def _binary_auc(self, y_true, y_score):
        if len(self.y_true) == 0:
            return 0.0
        
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)

        positive = y_score[y_true == 1]
        negative = y_score[y_true == 0]

        # edge case
        if len(positive) == 0 or len(negative) == 0:
            return 0.0

        # pairwise comparison
        comparisons = positive[:, None] > negative[None, :]
        ties = positive[:, None] == negative[None, :]

        return (comparisons.sum() + 0.5 * ties.sum()) / (len(positive) * len(negative))
     
    def auc(self):
        if len(self.y_true) == 0:
            return 0.0
        
        # Binary class
        if self.n_class == 2:
            return self._binary_auc(self.y_true, y_score)
        # Multi-class AUC
        else:
            y_true = np.asarray(self.y_true, dtype=float)
            y_score = np.asarray(self.y_score, dtype=float)
            
            n_class = self.n_class
            aucs = []

            for k in range(n_class):
                binary_true = (y_true == k).astype(int)
                auc_k = self._binary_auc(binary_true, y_score[:, k])

                if not np.isnan(auc_k):
                    aucs.append(auc_k)

            return np.mean(aucs) if len(aucs) >0 else 0.0
            
            
    
    def _rolling_cm(self):
        if self.rolling_cm is None:
            raise RuntimeError('You should input window_size first')
        
        cm = np.sum(self.rolling_cm, axis=0)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        
        
        accuracy = np.where(cm.sum()>0,np.diag(cm).sum() / cm.sum(),0.0)
        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        f1 = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)
        return accuracy, precision, recall, f1
    
    def rolling_auc(self):
        if self.window_size is None:
            raise RuntimeError('You should input window_size first')
        
        if len(self.rolling_true) == 0:
            return 0.0
        
        # Binary class
        if self.n_class == 2:
            return self._binary_auc(self.rolling_true, self.rolling_score)
        # Multi-class AUC
        else:
            y_true = np.asarray(self.rolling_true, dtype=float)
            y_score = np.asarray(self.rolling_score, dtype=float)
            
            n_class = self.n_class
            aucs = []

            for k in range(n_class):
                binary_true = (y_true == k).astype(int)
                auc_k = self._binary_auc(binary_true, y_score[:, k])

                if auc_k > 0:
                    aucs.append(auc_k)

            return np.mean(aucs) if len(aucs) >0 else 0.0
    
    def result(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        result = {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "auc":self.auc()
        }
        
        if self.window_size is not None:
            accuracy, precision, recall, f1 = self._rolling_cm()
            result.update({
            "rolling_accuracy": accuracy,
            "rolling_precision": precision,
            "rolling_recall": recall,
            "rolling_f1": f1,
            "rolling_auc":self.rolling_auc()
            })

        return result
    