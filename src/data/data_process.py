import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class CompressionDataset(Dataset):
    
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, index):

        origin_data = self.dataset[index]
        new_data = {}
        i = 0
        for key, value in origin_data.items():
            if i == 0:
                new_data["question"] = value
            elif i == 1:
                requirements = value
                requirements = requirements.split("; ")
                for j, requirement in enumerate(requirements):
                    new_data[f"requirement_{j+1}"] = requirement
            else:
                k = str(key[6])
                if int(k) == 6:
                    break
                new_data[f"demo_{k}"] = value
            i += 1
        return new_data 
                                    

class CompressionCommonDataset(Dataset):
    
    def __init__(self, dataset):
        
        self.dataset = dataset
    
    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, index):

        data = self.dataset[index]
        new_data = []
        for key, value in data.items():
            if "output" in key:
                k = str(key[6])
                if k==6: 
                    break
                new_data[f"demo_{k}"] = value
            else:
                new_data[key] = value
            
        return new_data
        
    

