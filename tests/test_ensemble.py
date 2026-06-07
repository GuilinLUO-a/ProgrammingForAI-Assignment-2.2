import numpy as np
import unittest
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier

class TestEnsembleClassifier(unittest.TestCase):
    def test_fit_train_bagging(self):
        """
        Test the fit and predict methods of the EnsembleClassifier using bagging.        
        """
        data = np.random.randint(0, 10, size=(100,5))
        X = data[:,:-1]
        y = data[:,-1].ravel()
        n_class= np.unique(y).size
        
        es = EnsembleClassifier(DecisionTreeClassifier, n_estimators=10,max_depth=5, min_samples_split=2,n_class=n_class)

        es.partial_fit(X, y)
        y_pred = es.predict(X)

        self.assertEqual(100, len(y_pred))
        
    def test_fit_train_random_forest(self):
        """
        Test the fit and predict methods of the EnsembleClassifier using random forest.
        """
        data = np.random.randint(0, 10, size=(100,5))
        X = data[:,:-1]
        y = data[:,-1].ravel()
        n_class= np.unique(y).size
        
        es = EnsembleClassifier(DecisionTreeClassifier, n_estimators=10,max_depth=5, min_samples_split=2,n_class=n_class, max_features=2)

        es.partial_fit(X, y)
        y_pred = es.predict(X)

        self.assertEqual(100, len(y_pred))

    def test_tree_number(self):
        """
        Test the number of trees in the ensemble.
        """
        es = EnsembleClassifier(DecisionTreeClassifier, n_estimators=10, max_depth=5, min_samples_split=2)
        
        self.assertEqual(10, len(es.trees))
        
    
if __name__ == '__main__':    
    unittest.main()