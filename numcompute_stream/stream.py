import numpy as np

class StreamTrainer:
    def __init__(self, pipeline, metrics):
        self.pipeline = pipeline
        self.metrics = metrics

        self.chunk_count = 0
        self.result = {}
        self.logs = []
        
    def fit_chunk(self, X, y):
        self.pipeline.partial_fit(X, y)
        self.chunk_count += 1
    
    def score_chunk(self, X, y):
        y_pred = self.pipeline.predict(X)

        self.metrics.update(y, y_pred)
        result = self.metrics.result()

        accuracy = np.mean(result['accuracy'])
        precision = np.mean(result['precision'])
        recall = np.mean(result['recall'])
        f1 = np.mean(result['f1'])
        
        self.logs.append({
            'chunk: ':self.chunk_count,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        
    
    def reset(self):
        self.chunk_count = 0
        self.logs = []