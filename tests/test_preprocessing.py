import numpy as np
import unittest
from numcompute_stream.stats import StreamingStats
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler

class TestPreprocess(unittest.TestCase):
    def test_imputer_numeric(self):
        X = np.array([1, 1, 1, 1, np.nan]).reshape(-1,1)
        
        num_imputer = Imputer(strategy='mean')
        num_imputer.partial_fit(X)
        X = num_imputer.transform(X)
        
        self.assertEqual(np.mean(X), 1.0)
        
    def test_imputer_mode(self):
        X = np.array(['good', 'good', 'good', 'good', '']).reshape(-1,1)
        
        mode_imputer = Imputer(strategy='mode')
        mode_imputer.partial_fit(X)
        X = mode_imputer.transform(X)
        
        self.assertFalse(np.any(X == ''))

    def test_one_hot_encoder(self):
        X = np.array(['good', 'good', 'good', 'good', 'bad']).reshape(-1,1)

        encoder = OneHotEncoder()
        
        encoder.partial_fit(X)
        X = encoder.transform(X)
        
        assert np.issubdtype(X.dtype, np.number)

    def test_standard_scaler(self):
        X = np.array([10000, 20000, 30000, 40000, 50000]).reshape(-1,1)
        
        scaler = StandardScaler()

        scaler.partial_fit(X)
        X = scaler.transform(X)
        max_number = max(X)
        self.assertLess(max_number, 10)
    
    def test_invalid_strategy_raise(self):
        with self.assertRaises(ValueError):
            test_imputer = Imputer(strategy='test')    
        
    def test_invalid_partial_fit_raise(self):
        X = np.array([])
        
        num_imputer = Imputer(strategy='mean')

        with self.assertRaises(ValueError):
            num_imputer.partial_fit(X)
            
    def test_transform_before_fit_raise(self):
        X = np.array([1, 1, 1, 1, np.nan]).reshape(-1,1)
        
        num_imputer = Imputer(strategy='mean')
        with self.assertRaises(RuntimeError):
            X = num_imputer.transform(X)
    
    def test_invalid_encoder_partial_fit_raise(self):
        X1 = np.array([])
        X2 = np.array(['1', '2', '3', '4'])
        X3 = np.array(['1', '2', '3', '4']).reshape(-1,1)
        X4 = np.array([[['1'],['2']],[['1'],['2']]])
        
        encoder = OneHotEncoder()
        encoder.partial_fit(X3)
        
        with self.assertRaises(ValueError):
            encoder.partial_fit(X1)
            
        with self.assertRaises(ValueError):
            encoder.partial_fit(X2)
            
        with self.assertRaises(ValueError):
            encoder.partial_fit(X4)
            
    def test_encoder_transform_before_fit_raise(self):
        X = np.array([1, 2, 3, 4]).reshape(-1,1)

        encoder = OneHotEncoder()
        with self.assertRaises(RuntimeError):
            encoder.transform(X)
            
    def test_invalid_scaler_partial_fit_raise(self):
        X1 = np.array([])
        X2 = np.array([1, 2, 3, 4])
        X3 = np.array([1, 2, 3, 4]).reshape(-1,1)
        X4 = np.array([[[1],[2]],[[1],[2]]])
        
        scaler = StandardScaler()
        scaler.partial_fit(X3)
        
        with self.assertRaises(ValueError):
            scaler.partial_fit(X1)
            
        with self.assertRaises(ValueError):
            scaler.partial_fit(X2)
            
        with self.assertRaises(ValueError):
            scaler.partial_fit(X4)
            
    def test_scaler_transform_before_fit_raise(self):
        X1 = np.array([])
        X2 = np.array([1, 2, 3, 4])
        X3 = np.array([1, 2, 3, 4]).reshape(-1,1)
        X4 = np.array([[[1],[2]],[[1],[2]]])

        scaler = StandardScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(X3)
            
        scaler.partial_fit(X3)
        
        with self.assertRaises(ValueError):
            scaler.partial_fit(X1)
            
        with self.assertRaises(ValueError):
            scaler.partial_fit(X2)
            
        with self.assertRaises(ValueError):
            scaler.partial_fit(X4)
        
if __name__ == '__main__':    
    unittest.main()