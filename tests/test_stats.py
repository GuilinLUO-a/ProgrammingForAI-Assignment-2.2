import numpy as np
import unittest
from numcompute_stream.stats import StreamingStats

class TestStreamingStats(unittest.TestCase):
    def test_mean_variance(self):
        X_chunk = np.array([[1, 1, 1, 1]])
        ss = StreamingStats()
        
        ss.update_stats(X_chunk)
        ss.update_stats(X_chunk)
        
        mean, variance = ss.get_meanVar()
        
        self.assertEqual(1.0, mean[0])
        self.assertEqual(0.0, variance[0])
        
    def test_get_quantiles(self):
        X_chunk = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        ss = StreamingStats()
        
        ss.update_stats(X_chunk)
        q50 = ss.get_quantiles(0.5)
        self.assertEqual(3.0, q50[0])
        
    def test_get_histograms(self):
        X_chunk = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        ss = StreamingStats()
        
        ss.update_stats(X_chunk)
        histogram = ss.get_histograms(bins=5)

        hist, edges = histogram[0]

        self.assertEqual(5, len(hist))
        self.assertEqual(6, len(edges))
    
    def test_rolling_quantiles(self):
        X_chunk = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        X_chunk2 = np.array([6, 7, 8, 9, 10]).reshape(-1, 1)
        ss = StreamingStats(window_size=2)
        
        ss.update_stats(X_chunk)
        ss.update_stats(X_chunk)
        ss.update_stats(X_chunk2)
        ss.update_stats(X_chunk2)
        
        q50 = ss.get_quantiles(0.5)
        self.assertEqual(8.0, q50[0])
    
    def test_rolling_histogram(self):
        X_chunk = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        X_chunk2= np.array([6, 7, 8, 9, 10]).reshape(-1, 1)
        ss = StreamingStats(window_size=2)
        
        ss.update_stats(X_chunk)
        ss.update_stats(X_chunk)
        ss.update_stats(X_chunk2)
        ss.update_stats(X_chunk2)
        histogram = ss.get_histograms(bins=5)

        hist, edges = histogram[0]

        self.assertEqual(5, len(hist))
        self.assertEqual(6, len(edges))
    
    def test_wrong_order_mean_raise(self):
        ss = StreamingStats()

        with self.assertRaises(RuntimeError):
            ss.get_meanVar()
            
    def test_wrong_order_hist_raise(self):
        ss = StreamingStats()

        with self.assertRaises(RuntimeError):
            ss.get_histograms()
    
    def test_wrong_quantiles_raise(self):
        X_chunk = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        ss = StreamingStats()
        
        ss.update_stats(X_chunk)

        with self.assertRaises(ValueError):
            ss.get_quantiles(-1)
    
    
    
if __name__ == '__main__':    
    unittest.main()