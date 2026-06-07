import numpy as np
import unittest
from numcompute_stream.metrics import StreamMetrics

class TestStreamMetrics(unittest.TestCase):
    def test_accuracy(self):
        """Test the accuracy method of StreamMetrics."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])    
            
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred)
        
        self.assertAlmostEqual(0.75, sm.accuracy())
        
    def test_accuracy_all_true(self):
        """Test the accuracy method when all predictions are correct."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0])
        
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred)
    
        self.assertAlmostEqual(1.0, sm.accuracy())

    def test_metrcis_result(self):
        """Test the result method of StreamMetrics."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])    
            
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred)
        
        result = sm.result()
        
        self.assertGreaterEqual(result['accuracy'], 0)
        self.assertLessEqual(result['accuracy'], 1)
        
        self.assertEqual(2, result['precision'].shape[0])
        self.assertEqual(2, result['recall'].shape[0])
        self.assertEqual(2, result['f1'].shape[0])

    def test_accumulate_cm(self):
        """"Test accumulative confusion matrix"""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0])    
            
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred)
        sm.update(y_true, y_pred)
        
        self.assertTrue(8, sm.n_samples)
        self.assertTrue(8, sm.n_correct)
        
    def test_auc(self):
        """Test auc of StreamMetrics."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0]) 
        y_score = np.array([0.1, 0.2, 0.3, 0.4])   
            
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred, y_score)
        
        auc = sm.auc()
        
        self.assertGreaterEqual(auc, 0)
        self.assertLessEqual(auc, 1)
        
    def test_rolling_auc(self):
        """Test rolling auc of StreamMetrics."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0]) 
        y_score = np.array([0.1, 0.2, 0.3, 0.4])   
            
        sm = StreamMetrics(n_class=2,window_size=2)
        sm.update(y_true, y_pred, y_score)
        sm.update(y_true, y_pred, y_score)
                
        auc = sm.rolling_auc()
        
        self.assertGreaterEqual(auc, 0)
        self.assertLessEqual(auc, 1)
        
    def test_reset(self):
        """Test the reset method of StreamMetrics."""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0])    
            
        sm = StreamMetrics(n_class=2)
        sm.update(y_true, y_pred)
        sm.reset()

        self.assertTrue(0==sm.n_samples)

    def test_input_wrong_shape(self):
        """Test raises ValueError for input wrong shape"""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 0])    
            
        sm = StreamMetrics(n_class=2)

        with self.assertRaises(ValueError):
            sm.update(y_true, y_pred)
            
    def test_mismatch_true_class(self):
        """Test raises ValueError for mismatch true class"""
        y_true = np.array([0, 1, 0, 2])
        y_pred = np.array([0, 1, 0, 0])    
            
        sm = StreamMetrics(n_class=2)

        with self.assertRaises(ValueError):
            sm.update(y_true, y_pred)
            
    def test_mismatch_pred_class(self):
        """Test raises ValueError for mismatch predictive class"""
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([2, 1, 0, -1])    
            
        sm = StreamMetrics(n_class=2)

        with self.assertRaises(ValueError):
            sm.update(y_true, y_pred)

    def test_rolling_cm_wrongly_used(self):
        """Test raises RuntimeError for wrongly used rolling confusion matrix"""
        sm = StreamMetrics(n_class=2)
        with self.assertRaises(RuntimeError):
            sm._get_rolling_cm()
            
    def test_rolling_auc_wrongly_used(self):
        """Test raises RuntimeError for wrongly used rolling AUC"""
        sm = StreamMetrics(n_class=2)
        with self.assertRaises(RuntimeError):
            sm.rolling_auc()
            
if __name__ == '__main__':    
    unittest.main()