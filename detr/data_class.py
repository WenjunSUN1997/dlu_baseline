from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader

class news_paper_dataset(Dataset):
    def __init__(self):
        self.file_list = self.get_fiel_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        path = '../data/'+self.file_list[item]
        return path

    def get_fiel_list(self):
        file_list = os.listdir('../data')
        file_list = list(set(x.split('.')[0] for x in file_list))
        return file_list




def get_data_loader(batch_size: int):
    return DataLoader(news_paper_dataset(), batch_size=batch_size)

data_loader = get_data_loader(2)
for index, infor in enumerate(data_loader):
    print(index)
    print(infor)