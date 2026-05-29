class StreamMetrics:
    
    def __init__(self, window_size=None):
        ...

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        ...

    def reset(self):
        ...

    def result(self):
        return {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "auc": ...
        }

    def window_result(self):
        return {
            "accuracy_window": ...,
            "precision_window": ...,
            
        }