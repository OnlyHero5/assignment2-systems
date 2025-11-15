import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self, data:List[int], batch_size:int, context_length:int, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle
        self.data_length = len(data)

    def get_train_batch_data(self):
        idxs = np.random.randint(0, self.data_length - self.context_length-1, size=(self.batch_size,))
        x = np.stack([self.data[idx:idx+self.context_length] for idx in idxs])
        y = np.stack([self.data[idx+1:idx+self.context_length+1] for idx in idxs])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def get_valid_batch_data(self):
       
     start_num = (self.data_length - self.context_length - 1) // self.batch_size
     for i in range(start_num):
        bias = i * self.batch_size
        x = np.stack([self.data[bias:bias+self.context_length] for i in range(self.batch_size)])
        y = np.stack([self.data[bias+1:bias+self.context_length+1] for i in range(self.batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return self.data_length // self.batch_size