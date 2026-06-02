import matplotlib.pyplot as plt
import numpy as np

def plot_metric_over_time(metric_values, title, ylabel, save_path=None):
    plt.figure()
    x = np.arange(1, len(metric_values)+1)
    y = np.asarray(metric_values)
    
    plt.plot(x, y, color='blue', marker='.')
    plt.title(title)
    plt.xlabel('Chunk')
    plt.ylabel(ylabel)
    plt.grid()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()
    
    plt.close()
    
def compare_models(metric1, metric2, labels, save_path=None):
    plt.figure()
    y1 = np.asarray(metric1)
    y2 = np.asarray(metric2)
    x = np.arange(1, max(len(y1)+1, len(y2)+1))
    
    plt.plot(x, y1, color='blue', marker='.',label=labels[0])
    plt.plot(x, y2, color='orange', marker='.',label=labels[1])
    plt.title('Compare two metrics')
    plt.xlabel('Chunk')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()
    
    plt.close()
    

def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None):
    plt.figure()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    x = np.arange(1, max(len(y_true)+1, len(y_pred)+1))

    plt.scatter(x, y_true, c='blue', marker='o', label='True')
    plt.scatter(x, y_pred, c='red', marker='o', label='Predictive')

    plt.title('Predictions vs Ground truth')
    plt.xlabel('Sample')
    plt.ylabel('Class')
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()
    
    plt.close()
    


    
