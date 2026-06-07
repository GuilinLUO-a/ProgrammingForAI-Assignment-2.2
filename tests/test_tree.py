import numpy as np
import unittest
from numcompute_stream.tree import DecisionTreeClassifier

class TestDecisionTreeClassifier(unittest.TestCase):
    def test_fit_train(self):
        """Test fit and train decision tree classifier"""
        data = np.random.randint(0, 10, size=(100,5))
        X = data[:,:-1]
        y = data[:,-1].ravel()
        n_class= np.unique(y).size

        tree = DecisionTreeClassifier(max_depth=2, min_samples_split=2, n_class=n_class)
        tree.partial_fit(X, y)
        y_pred = tree.predict(X)
        
        self.assertEqual(100, y_pred.size)
        
    def test_invalid_input_raise(self):
        """Test raise ValueError for invalid input"""
        data = np.random.randint(0, 10, size=(100,5))
        X = data[:,:-1]
        y = data[:,-1].ravel()
        n_class= np.unique(y).size

        with self.assertRaises(ValueError):
            tree = DecisionTreeClassifier(max_depth=2, min_samples_split=2, n_class=n_class, criterion='Test')

    def test_tree_depth_predictive(self):
        """Test tree depth and predictive"""
        X = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1]).ravel()
        
        tree = DecisionTreeClassifier(max_depth=10, min_samples_split=1, n_class=2)
        tree.partial_fit(X, y)
        y_pred = tree.predict(X)
        
        acc = np.mean(y_pred == y)
        self.assertAlmostEqual(1.0, acc)
        
    def test_predict_no_fit(self):
        """Test get 0.0 for predict without fit"""
        X = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)

        tree = DecisionTreeClassifier(max_depth=10, min_samples_split=1, n_class=2)
        y_pred = tree.predict(X)
        
        acc = y_pred.sum()
        self.assertEqual(0.0, acc)
    
if __name__ == '__main__':    
    unittest.main()