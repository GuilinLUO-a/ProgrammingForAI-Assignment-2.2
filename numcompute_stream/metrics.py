import numpy as np
from collections import deque

class StreamMetrics:
    def __init__(self, n_classes = 2, window_size=None):
        self.n_class = n_classes
        self.window_size = window_size
        self.cm = None # Confusion matrix
        self.n_correct = None
        self.n_samples = None
        self.rolling_cm = None

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        if self.cm is None:
            self.reset()
            
        y_true_chunk = np.asarray(y_true_chunk)
        y_pred_chunk = np.asarray(y_pred_chunk)
        
        if y_true_chunk.shape != y_pred_chunk.shape:
            raise ValueError("The shape of two arrays are not the same")    

        cm = np.bincount(self.n_class * y_true_chunk + y_pred_chunk, 
                         minlength= self.n_class * self.n_class).shape(self.n_class, self.n_class)

        self.cm += cm

        n_correct = np.diag(cm).sum() # The number of correct samples
        n_samples = cm.sum()          # The number of all samples

        self.n_correct += n_correct
        self.n_samples += n_samples
        
        if self.window_size is not None:
            self.rolling_cm.append(cm)

    def reset(self):
        self.cm = np.zeros((self.n_class, self.n_class), dtype=np.int64)
        self.n_correct = 0
        self.n_samples = 0
        
        if self.window_size is not None:
            self.rolling_cm = deque(maxlen=10)

        
        
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
    
    def _rolling_cm(self):
        if self.rolling_cm is None:
            raise RuntimeError('You should input window_size first')
        
        cm = self.rolling_cm
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        
        
        accuracy = np.where(cm.sum()>0,np.diag(cm).sum() / cm.sum(),0.0)
        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        f1 = np.where((precision + recall) > 0,
                         2 * precision * recall / (precision + recall), 0.0)
        return accuracy, precision, recall, f1
    
    def result(self):
        if self.cm is None:
            raise RuntimeError('You should call update() first')
        
        result = {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }
        
        if self.window_size is not None:
            accuracy, precision, recall, f1 = self._rolling_cm()
            result = {
            "rolling_accuracy": accuracy,
            "rolling_precision": precision,
            "rolling_recall": recall,
            "rolling_f1": f1,
            }

        return result
    