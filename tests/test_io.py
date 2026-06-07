import numpy as np
import unittest
from numcompute_stream.io import load_csv, load_stream_csv

class TestLoadCsv(unittest.TestCase):
    def test_load_file(self):
        """Test the load_csv function."""
        data_path = 'data/adult.csv'
        X, y = load_csv(file_path=data_path)
        
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
    
    def test_invalid_file_path_raise(self):
        """Test raises FileNotFoundError invalid file path."""
        with self.assertRaises(FileNotFoundError):
            load_csv(file_path='test/test')
            
class TestLoadStreamCSV(unittest.TestCase):
    def test_load_stream_csv(self):
        """Test the load_stream_csv function."""
        data_path = 'data/adult.csv'
        stream_chunk = load_stream_csv(data_path, chunk_size=100)
        X_chunk, y_chunk = next(stream_chunk)
        
        self.assertGreater(len(X_chunk), 0)
        self.assertGreater(len(y_chunk), 0)
        
    def test_invalid_file_path_raise(self):
        """Test raises FileNotFoundError for invalid file path."""
        with self.assertRaises(FileNotFoundError):
             next(load_stream_csv(file_path='test/test', chunk_size=100))

if __name__ == '__main__':    
    unittest.main()