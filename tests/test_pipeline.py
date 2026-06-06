import numpy as np
import unittest
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import PipeLine, Preprocessor, LabelEncoder

class TestPreprocessor(unittest.TestCase):
    def test_partial_fit(self):
        X = np.array([
            [1,'good','HD',2],
            [2,'bad','D',2],
            [3,'bad','HD',2]
        ])
        prepro = Preprocessor()
        prepro.partial_fit(X)
        X = prepro.transform(X)

        self.assertEqual(X.shape[0], 3)
    
    def test_reset(self):
        X = np.array([
            [1,'good','HD',2],
            [2,'bad','D',2],
            [3,'bad','HD',2]
        ])
        prepro = Preprocessor()
        prepro.partial_fit(X)
        prepro.reset()
        
        self.assertTrue(prepro.num_cols is None)

    def test_invalid_fit_raise(self):
        X = np.array([
        ])
        
        prepro = Preprocessor()
        
        with self.assertRaises(ValueError):
            prepro.partial_fit(X)
            
    def test_invalid_transform_order_raise(self):
        X = np.array([
            [1,'good','HD',2],
            [2,'bad','D',2],
            [3,'bad','HD',2]
        ])
        
        prepro = Preprocessor()
        
        with self.assertRaises(RuntimeError):
            prepro.transform(X) 
            
class TestLabelEncoder(unittest.TestCase):
    def test_partial_fit(self):
        y = np.array(['good', 'bad','good', 'bad'])

        le = LabelEncoder()
        le.partial_fit(y)

        class_to_idx = le.class_to_idx
        idx_to_class = le.idx_to_class
        
        self.assertTrue(len(class_to_idx), 4)
        self.assertTrue(len(idx_to_class), 4)

    def test_transform(self):
        y = np.array(['good', 'bad','good', 'bad'])

        le = LabelEncoder()
        le.partial_fit(y)
        
        y_transform = le.transform(np.array(['good','good']))
        self.assertIn(0, y_transform)
        
    def test_inverse_transform(self):
        y = np.array(['good', 'bad','good', 'bad'])

        le = LabelEncoder()
        le.partial_fit(y)
        
        y_inverse_transform = le.inverse_transform(np.array([0,0]))
        self.assertIn('good', y_inverse_transform)

class TestPipeLine(unittest.TestCase):
    def test_partial_fit(self):
        pipe = PipeLine([
            ('preprocessor',Preprocessor()),
            ('model',DecisionTreeClassifier(max_depth=5, min_samples_split=2))
        ])
        X = np.array([
            [1,'good','HD',2],
            [2,'bad','D',2],
            [3,'bad','HD',2]
        ])
        y = np.array(['Succeed','Fail','Fail'])
        pipe.partial_fit(X, y)
        y_pred = pipe.predict(X)
        
        self.assertEqual(len(y_pred), 3)
        for p in y_pred:
            self.assertIn(p, ['Succeed','Fail'])
        
if __name__ == '__main__':    
    unittest.main()