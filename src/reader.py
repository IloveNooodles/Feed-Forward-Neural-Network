import pandas as pd
from sklearn import datasets

class Reader:
    def __init__(self) -> None:
        self.filepath = None
        self.df = None
    
    def read_excel(self, filepath: str):
        self.filepath = filepath
        df = pd.read_excel(filepath, index_col=False)
        return df