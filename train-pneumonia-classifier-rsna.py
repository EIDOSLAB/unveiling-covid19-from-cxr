# %%
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

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from utils.config import base_path
from utils import vars
from utils import trainer
from utils import metrics
from utils import utils

import matplotlib.pyplot as plt
import seaborn as sns

from models import covid_classifier
from models import pneumonia_classifier

from datasets import corda
from datasets import rsna

import functools

# %%
lr = 1e-1
seed = vars.seed
device = torch.device('cuda:0')
epochs = 1
metric = 'auc'
mode = 'max'

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=160, help='num. of epochs')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate default (0.1)')
parser.add_argument('--metric', type=str, default='loss', help='Metric for best model')
parser.add_argument('--mode', type=str, default='min', help='max or min')
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
metric = args.metric
mode = args.mode

# %%
device = torch.device('cuda:0')
utils.set_seed(seed)

# DATA PREPROCESSING
use_preprocessed = True
corda_dataset = 'CORDA-dataset'
rsna_dataset = 'rsna_bal_subset'
model_preprocessed = 'not-equalized'
feature_preprocessed = ''
preprocessed = ''

if use_preprocessed:
    corda_dataset = 'CORDA-dataset-v4-equalized+masked'
    rsna_dataset = 'rsna_bal_subset-equalized+masked'
    model_preprocessed = 'equalized'
    feature_preprocessed = '-eq+masked'
    preprocessed = '(preprocessed)'

# %%
corda_version = f'CORDA-dataset-{vars.corda_version}'
corda_basepath = os.path.join(base_path, 'corda', corda_version, corda_dataset)

# %%
corda_df = pd.read_csv(os.path.join(corda_basepath, 'CORDA_fix.csv'))
corda_df.groupby('covid').count()

# %%
corda_train_df, corda_test_df = train_test_split(corda_df, test_size=0.3, random_state=vars.seed, stratify=corda_df.covid)
corda_train_df.groupby('covid').count()

# %%
rsna_basepath = os.path.join(base_path, rsna_dataset)

rsna_train_df = pd.read_csv(os.path.join(rsna_basepath, 'stage_2_train_labels_subset.csv'))
rsna_train_df = corda.preprocess_rsna_df(rsna_train_df)

rsna_train_df, rsna_test_df = train_test_split(rsna_train_df, test_size=0.3, random_state=vars.seed, stratify=rsna_train_df.label)
rsna_train_df, rsna_val_df = train_test_split(rsna_train_df, test_size=0.2, random_state=vars.seed, stratify=rsna_train_df.label)

# %%
rsna_train_df.groupby('label').count()

# %%
rsna_val_df.groupby('label').count()

# %%
rsna_test_df.groupby('label').count()

# %%
train_df = pd.concat((corda_train_df, rsna_train_df))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
])

train_dataset = corda.CORDARSNA(
    train_df, rsna_base_path=rsna_basepath,
    corda_base_path=corda_basepath, transform=transforms
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10,
    shuffle=False, num_workers=10,
    worker_init_fn=lambda id: utils.set_seed(seed),
    pin_memory=True
)

mean, std = utils.get_mean_and_std(train_dataloader)
print(f'Mean & std for corda+rsna:', mean, std)

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
train_dataset = rsna.RSNA(rsna_train_df, rsna_basepath, transform=train_transforms)
val_dataset = rsna.RSNA(rsna_val_df, rsna_basepath, transform=transforms)
test_dataset = rsna.RSNA(rsna_test_df, rsna_basepath, transform=transforms)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10,
    shuffle=True, num_workers=4,
    worker_init_fn=lambda id: utils.set_seed(vars.seed)
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# %%
print(f'using lr {lr}')

# %%
model = pneumonia_classifier.PneumoniaClassifierRSNA50(pretrained=True).to(device)

# %%
name = f'resnet50-pneumonia-classifier-s{seed}-rsna{feature_preprocessed}'

# %%
#checkpoint = torch.load(f'models/{vars.corda_version}/{name}.pt')
#model.load_state_dict(checkpoint['model'])
#print(f'Restoring training from epoch {checkpoint["epoch"]}')

# %%
criterion = functools.partial(F.binary_cross_entropy, reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

optimizer.load_state_dict(checkpoint['optimizer'])

# %%
tracked_metrics = [
    metrics.Accuracy(),
    metrics.RocAuc(),
    metrics.FScore()
]

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
)

# %%
print('best model: ')
test_logs, test_cm = trainer.test(
    model=best_model, test_dataloader=test_dataloader,
    criterion=criterion, metrics=tracked_metrics, device=device,
)

labels = ['NORMAL', 'PNEUMONIA']
ax = sns.heatmap(
    test_cm.get(normalized=True), annot=True, fmt=".2f",
    xticklabels=labels, yticklabels=labels
)
hm = ax.get_figure()
ax.set_title(f'RSNA {preprocessed} (best)')
plt.xlabel('predicted')
plt.ylabel('ground')
hm.savefig(f'logs/{vars.corda_version}/{name}/best.png')
hm.clf()
