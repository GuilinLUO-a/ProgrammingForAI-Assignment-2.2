import numpy as np

def load_csv(fileName='',delimeter=',',skipheader=1):
    try:
        data = np.genfromtxt(
            fname=fileName,
            delimiter=delimeter,
            skip_header=skipheader,
            dtype=str
        )
        
        if data.size == 0:
            raise ValueError(f"{fileName}"" is empty")
        
        X = data[:,:-1]
        y = data[:,-1]
        return X, y
    
    except FileNotFoundError:
        print('There is no specific file')
        raise FileNotFoundError(f"{fileName}"" doesn't exist")

X, y = load_csv(fileName='adult.csv')
print(X.shape)
print(y.shape)
