#%%
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import random
import os
import copy
import PIL
import argparse
import functools

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils.config import base_path
from utils import trainer
from utils import metrics
from utils import utils
from utils import vars

from datasets import chestxray
from datasets import corda
from models import pneumonia_classifier

import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

# %%
device = torch.device('cuda:0')
seed = vars.seed
utils.set_seed(seed)

# %%
corda_dataset = 'CORDA-dataset-v4-equalized+masked'
corda_version = f'CORDA-dataset-{vars.corda_version}'
corda_basepath = os.path.join(base_path, 'corda', corda_version, corda_dataset)

# %%
corda_df = pd.read_csv(os.path.join(corda_basepath, 'CORDA_fix.csv'))
corda_train_df, corda_test_df = train_test_split(corda_df, test_size=0.3, random_state=vars.seed, stratify=corda_df.covid)

# %%
chestxray_dataset = 'chest_xray-equalized+masked'
dataset_path = os.path.join(base_path, chestxray_dataset)

# %%
train_df = pd.read_csv(os.path.join(dataset_path, 'train_3_classes.csv'))
val_df = pd.read_csv(os.path.join(dataset_path, 'val_3_classes.csv'))
test_df = pd.read_csv(os.path.join(dataset_path, 'test_3_classes.csv'))

# %%
train_df = chestxray.preprocess_chest_df(train_df)
val_df = chestxray.preprocess_chest_df(val_df)
test_df = chestxray.preprocess_chest_df(test_df)

# %%
train_df.groupby('label').count()
val_df.groupby('label').count()

# %%
train_df['covid'] = 0
val_df['covid'] = 0
stats_train_df = pd.concat((corda_train_df, train_df, val_df))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
])

train_dataset = corda.CORDAChestXRay(
    stats_train_df, chest_base_path=dataset_path,
    corda_base_path=corda_basepath, transform=transforms
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10,
    shuffle=False, num_workers=10,
    worker_init_fn=lambda id: utils.set_seed(seed),
    pin_memory=True
)

mean, std = utils.get_mean_and_std(train_dataloader)
print(f'Mean & std for corda+chestxray:', mean, std)


# %%
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomAffine(0, translate=(0, 0.1), scale=(1, 1.10)),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

# %%
train_dataset = chestxray.ChestXRay(train_df, dataset_path, transform=train_transforms)
val_dataset = chestxray.ChestXRay(val_df, dataset_path, transform=transforms)
test_dataset = chestxray.ChestXRay(test_df, dataset_path, transform=transforms)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10,
    shuffle=True, num_workers=4,
    worker_init_fn=lambda id: utils.set_seed(vars.seed)
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# %%
lr = 1e-2
epochs = 150
metric = 'loss'
mode = 'min'

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150, help='num. of epochs (default 150)')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate default (1e-2)')
parser.add_argument('--metric', type=str, default='loss', help='Metric for best model')
parser.add_argument('--mode', type=str, default='min', help='max or min')
parser.set_defaults(unfreeze_conv=False)
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
metric = args.metric
mode = args.mode
print(f'using lr {lr}')

# %%
model = pneumonia_classifier.PneumoniaClassifierChest(pretrained=True).to(device)

# %%
criterion = functools.partial(F.cross_entropy, reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

# %%
tracked_metrics = [
    metrics.Accuracy(multiclass=True),
]

name = f'resnet18-pneumonia-classifier-s{seed}-3-classes-eq+masked'
utils.ensure_dir(f'logs/{vars.corda_version}/{name}')

# %%
best_model = trainer.fit(
    model=model, train_dataloader=train_dataloader,
    val_dataloader=val_dataloader, test_dataloader=test_dataloader,
    test_every=10, criterion=criterion,
    optimizer=optimizer, scheduler=lr_scheduler,
    metrics=tracked_metrics, n_epochs=epochs,
    name=name,
    metric_choice=metric, mode=mode, device=device,
    multiclass=True
)

# %%
print('best model: ')
test_logs, test_cm = trainer.test(
    model=best_model, test_dataloader=test_dataloader,
    criterion=criterion, metrics=tracked_metrics, device=device,
    multiclass=True
)
ax = sns.heatmap(test_cm.get(normalized=True), annot=True, fmt=".2f")
hm = ax.get_figure()
ax.set_title('Chest X Ray (preprocessed)')
plt.xlabel('predicted')
plt.ylabel('ground')
hm.savefig(f'logs/{vars.corda_version}/{name}/best.png')
hm.clf()
