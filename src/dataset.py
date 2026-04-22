import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CIFAKEDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataframe(path):
    data = []
    for split in ['test', 'train']:
        split_path = os.path.join(path, split)
        if os.path.exists(split_path):
            for label in ['FAKE', 'REAL']:
                label_path = os.path.join(split_path, label)
                if os.path.exists(label_path):
                    with os.scandir(label_path) as entries:
                        for entry in entries:
                            if entry.is_file():
                                data.append({
                                    'file_path': entry.path,
                                    'label': 1 if label == 'FAKE' else 0,
                                    'split': split
                                })
    return pd.DataFrame(data)
