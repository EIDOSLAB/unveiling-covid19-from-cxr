import pandas as pd
import torch
import os
import PIL

def preprocess_chest_df(df):
    df['covid'] = 0
    df['dataset'] = 'chest_xray'

    df.loc[df.label == 2, 'label'] = 1

    df['rx'] = 0
    df.loc[df.label == 1, 'rx'] = 1

    return df

class CORDAChestXRay(torch.utils.data.dataset.Dataset):
    def __init__(self, df, chest_base_path, corda_base_path, transform=None):
        super().__init__()

        self.df = df
        self.chest_base_path = chest_base_path
        self.corda_base_path = corda_base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def df2path(self, image_id, covid, rx):
        covid_map = {0: 'COVID-', 1: 'COVID+'}
        rx_map = {0: 'RX-', 1: 'RX+'}

        return os.path.join(f'{rx_map[rx]}-{covid_map[covid]}/', image_id + '.png')

    def __getitem__(self, index):
        entry = self.df.iloc[index]

        fname = entry.image_id
        if entry.dataset == 'CORDA-dataset':
            fname = self.df2path(fname, entry.covid, entry.rx)
            fname = os.path.join(self.corda_base_path, fname)

        elif entry.dataset == 'chest_xray':
            fname = os.path.join(self.chest_base_path, fname)

        elif entry.dataset == 'covid-chestxray-dataset':
            fname = os.path.join('images', fname)

        img = PIL.Image.open(fname).convert('L')
        img = self.transform(img)

        return img, int(entry.covid), entry.image_id

def preprocess_rsna_df(df):
    df['covid'] = 0
    df['dataset'] = 'rsna'
    df['rx'] = 0
    df.loc[df.label == 1, 'rx'] = 1
    return df

def preprocess_cohen_df(df):
    df['dataset'] = 'cohen'
    df['rx'] = 1
    return df

class CORDA(torch.utils.data.dataset.Dataset):
    def __init__(
        self, df, corda_base_path=None,
        chest_base_path=None, rsna_base_path=None,
        cohen_base_path=None, transform=None
    ):
        super().__init__()

        self.df = df
        self.corda_base_path = corda_base_path
        self.chest_base_path = chest_base_path
        self.rsna_base_path = rsna_base_path
        self.cohen_base_path = cohen_base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def df2path(self, image_id, covid, rx):
        covid_map = {0: 'COVID-', 1: 'COVID+'}
        rx_map = {0: 'RX-', 1: 'RX+'}

        return os.path.join(f'{rx_map[rx]}-{covid_map[covid]}/', image_id + '.png')

    def __getitem__(self, index):
        entry = self.df.iloc[index]

        fname = entry.image_id
        if entry.dataset == 'CORDA-dataset':
            fname = self.df2path(fname, entry.covid, entry.rx)
            fname = os.path.join(self.corda_base_path, fname)

        elif entry.dataset == 'chest_xray':
            fname = os.path.join(self.chest_base_path, fname)

        elif entry.dataset == 'rsna':
            fname = os.path.join(self.rsna_base_path, 'train', fname + '.png')

        elif entry.dataset == 'cohen':
            fname = os.path.join(self.cohen_base_path, 'images', fname)

        img = PIL.Image.open(fname).convert('L')
        img = self.transform(img)

        return img, int(entry.covid), entry.image_id
