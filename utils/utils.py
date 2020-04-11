import torch
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def get_mean_and_std(dataloader):
    num_samples = 0.
    mean = 0.
    std = 0.
    for batch, _, _ in tqdm(dataloader):
        batch_size = batch.size(0)
        batch = batch.view(batch_size, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        num_samples += batch_size

    mean /= num_samples
    std /= num_samples

    return mean, std
