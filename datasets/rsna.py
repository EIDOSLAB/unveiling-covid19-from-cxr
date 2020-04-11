import pandas as pd
import torch
import os
import PIL

def preprocess_rsna_df(df):
    df['dataset'] = 'rsna'
    return df

class RSNA(torch.utils.data.dataset.Dataset):
    def __init__(self, df, rsna_base_path, transform=None):
        super().__init__()

        self.df = df
        self.rsna_base_path = rsna_base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]

        fname = entry.image_id
        fname = os.path.join(self.rsna_base_path, 'train', fname + '.png')
        img = PIL.Image.open(fname).convert('L')
        img = self.transform(img)

        return img, int(entry.label), entry.image_id
