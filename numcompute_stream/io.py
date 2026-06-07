import numpy as np
import csv

def load_csv(file_path='',delimiter=',',skipheader=1):
    try:
        data = np.genfromtxt(
            fname=file_path,
            delimiter=delimiter,
            skip_header=skipheader,
            dtype=str
        )
        
        if data.size == 0:
            raise ValueError(f"{file_path}"" is empty")
        
        X = data[:,:-1]
        y = data[:,-1]
        return X, y
    
    except FileNotFoundError:
        print('There is no specific file')
        raise FileNotFoundError(f"{file_path}"" doesn't exist")

def load_stream_csv(file_path='', chunk_size=1000):
    '''
    Load a CSV file in chunks and yield the features and labels for each chunk.
    '''
    try:
        chunk = []
        with open(file_path,'r') as f:
            lines = csv.reader(f)
            next(lines)

            for line in lines:
                chunk.append(line)

                if len(chunk) == chunk_size:
                    data_chunk = np.asarray(chunk)

                    X = data_chunk[:,:-1]
                    y = data_chunk[:,-1]
                    
                    chunk = []
                    
                    yield X, y
            if chunk:
                data_chunk = np.asarray(chunk)

                X = data_chunk[:,:-1]
                y = data_chunk[:,-1]
                
                yield X, y

    except FileNotFoundError:
        print('There is no specific file')
        raise FileNotFoundError(f"{file_path}"" doesn't exist")


