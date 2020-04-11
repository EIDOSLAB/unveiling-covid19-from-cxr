import torch
import PIL
import os

def preprocess_chest_df(df):
    df['dataset'] = 'chest_xray'
    return df

class ChestXRay(torch.utils.data.dataset.Dataset):
    def __init__(self, df, base_path, transform=None):
        super().__init__()

        self.df = df
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fname, label, dataset = self.df.iloc[index].image_id, self.df.iloc[index].label, self.df.iloc[index].dataset
        img = PIL.Image.open(os.path.join(self.base_path, fname)).convert('L')
        img = self.transform(img)

        return img, label, fname
