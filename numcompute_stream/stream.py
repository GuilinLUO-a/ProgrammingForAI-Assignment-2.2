import numpy as np
import time
import tracemalloc

class StreamTrainer:
    def __init__(self, pipeline, metrics):
        self.pipeline = pipeline
        self.metrics = metrics

        self.chunk_count = 0
        self.training_time = []
        self.memory = []
        self.logs = []
        
    def fit_chunk(self, X, y):
        tracemalloc.start()
        start = time.time()
        
        self.pipeline.partial_fit(X, y)
        self.chunk_count += 1
        
        end = time.time()
        _, peak1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.training_time.append(round(end - start, 4))
        self.memory.append(round(peak1 / 1024 / 1024, 4))
    
    def score_chunk(self, X, y):
        if self.chunk_count == 0:
            return self
        
        y_pred = self.pipeline.predict(X)

        le = self.pipeline.label_encoder
        y_pred = le.transform(y_pred)
        y = le.transform(y)

        self.metrics.update(y, y_pred)
        result = self.metrics.result()
        
        self.logs.append({
            'chunk':self.chunk_count,
            "accuracy": result['accuracy'],
            "precision": np.mean(result['precision']),
            "recall": np.mean(result['recall']),
            "f1": np.mean(result['f1']),
            "training_time(senconds)":self.training_time[-1],
            "memory(mb)":self.memory[-1]
        })
        
        if 'auc' in result:
            self.logs[-1].update({"auc":np.mean(result['auc'])})
        
        if 'rolling_accuracy' in result:
            self.logs[-1].update({
                "rolling_accuracy": result['rolling_accuracy'],
                "rolling_precision": np.mean(result['rolling_precision']),
                "rolling_recall": np.mean(result['rolling_recall']),
                "rolling_f1": np.mean(result['rolling_f1'])
            })
            
        if 'rolling_auc' in result:
            self.logs[-1].update({"rolling_auc":np.mean(result['rolling_auc'])})
    
    def reset(self):
        self.chunk_count = 0
        self.training_time = []
        self.memory = []
        self.logs = []