import torch
import copy
from . import vars
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from .metrics import ConfusionMatrix
from tqdm import tqdm

def summarize_metrics(metrics):
    return ' - '.join(list(map(lambda kv: '{}: {:.4f}'.format(kv[0], float(str(kv[1]))), metrics.items())))

def report_metrics(metrics, end='\n'):
    print(summarize_metrics(metrics), end=end, flush=True)

def run(model, dataloader, criterion, optimizer, metrics, phase, device=torch.device('cuda:0'), weight=None, self_supervised=False, multiclass=False):
    num_batches = 0.
    loss = 0.

    cm = ConfusionMatrix(multiclass=multiclass)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    for metric in metrics:
        metric.reset()

    for data, labels, _ in tqdm(dataloader, desc=phase, leave=False):
        data, labels = data.to(device), labels.to(device)

        running_loss = 0.
        output = None

        with torch.set_grad_enabled(phase == 'train'):
            if weight is not None:
                weight = weight[labels]

            output = model(data)
            if not self_supervised:
                if not multiclass:
                    output = output.view(-1)
                    labels = labels.float()

                running_loss = criterion(output, labels, weight=weight)
            else:
                running_loss = criterion(output, data)

        if phase == 'train':
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.zero_grad()
            else:
                for opt in optimizer:
                    opt.zero_grad()

            running_loss.backward()

            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.step()
            else:
                for opt in optimizer:
                    opt.step()

        if multiclass:
            _, output = torch.max(output, 1)

        if not self_supervised:
            cm.accumulate(output, labels)

        for metric in metrics:
            metric.accumulate(output, labels)

        loss += running_loss.item()
        num_batches += 1

    logs = { metric.__name__: copy.deepcopy(metric) for metric in metrics }
    logs.update({'loss': loss / num_batches})
    return logs, cm

def make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params=None):
    checkpoint = {
        'epoch': epoch, 'model': model.state_dict(), 'metrics': metrics
    }

    if isinstance(optimizer, torch.optim.Optimizer):
        checkpoint.update({'optimizer': optimizer.state_dict()})
    else:
        checkpoint.update({'optimizers': []})
        for opt in optimizer:
            checkpoint['optimizers'].append(opt.state_dict())

    if checkpoint_params is not None:
        checkpoint.update(checkpoint_params)
    return checkpoint

def is_better(a, b, mode='min'):
    if mode == 'min':
        return a < b
    elif mode == 'max':
        return a > b

    return False

def plot_losses(train, val, test, name):
    df = pd.DataFrame({'train': train, 'val': val, 'test': test})
    ax = sns.lineplot(data=df)
    ax.set_title(name)
    hm = ax.get_figure()
    hm.savefig(f'logs/{vars.corda_version}/{name}/loss.png')
    hm.clf()

def save_cm(cm, title, path, filename, epoch=None, normalized=False, format="d"):
    ax = sns.heatmap(cm.get(normalized=normalized), annot=True, fmt=".2f")
    ax.set_title(f'{title}')
    if epoch is not None:
        ax.set_title(f'{title} epoch {epoch}')
    plt.xlabel('predicted')
    plt.ylabel('ground')
    hm = ax.get_figure()
    hm.savefig(f'logs/{vars.corda_version}/{path}/{filename}.png')
    hm.clf()


def fit(model, train_dataloader, val_dataloader, test_dataloader, test_every,
        criterion, optimizer, scheduler, metrics, n_epochs, name,
        weight={'train': None, 'val': None, 'test': None}, self_supervised=False, multiclass=False,
        metric_choice='loss', mode='min', device=torch.device('cuda:0'), checkpoint_params=None, callbacks=None):

    best_metric = 0.
    best_model = None

    train_losses = []
    val_losses = []
    test_losses = []

    test_logs = {'loss': 1.}

    for epoch in range(n_epochs):

        train_logs, train_cm = run(
            model=model, dataloader=train_dataloader,
            criterion=criterion, weight=weight['train'], optimizer=optimizer,
            metrics=metrics, phase='train', device=device, self_supervised=self_supervised,
            multiclass=multiclass
        )
        save_cm(train_cm, f'{name}-train', name, 'train', epoch)

        if callbacks is not None and callbacks['train'] is not None:
            callbacks['train']()

        val_logs, val_cm = run(
            model=model, dataloader=val_dataloader,
            criterion=criterion, weight=weight['val'], optimizer=None,
            metrics=metrics, phase='val', device=device, self_supervised=self_supervised,
            multiclass=multiclass
        )
        save_cm(val_cm, f'{name}-val', name, 'val', epoch)

        if callbacks is not None and callbacks['val'] is not None:
            callbacks['val']()

        print(f'Epoch: {epoch:03d} | VAL ', end='')
        report_metrics(val_logs, end=' | TRAIN ')
        report_metrics(train_logs, end=' |\n')

        if scheduler is not None:
            scheduler.step(val_logs['loss'])

        torch.save(make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params), f'models/{vars.corda_version}/{name}.pt')

        if best_model is None or is_better(float(str(val_logs[metric_choice])), best_metric, mode):
            best_metric = float(str(val_logs[metric_choice]))
            best_model = copy.deepcopy(model)
            torch.save(make_checkpoint(epoch, model, optimizer, metrics, checkpoint_params), f'models/{vars.corda_version}/{name}-best.pt')

        if (epoch+1) % test_every == 0:
            test_logs, test_cm = test(
                model=model, test_dataloader=test_dataloader,
                criterion=criterion, metrics=metrics,
                self_supervised=self_supervised, device=device,
                multiclass=multiclass, weight=weight['test']
            )

            save_cm(test_cm, f'{name}-test', name, 'test', epoch)

            if callbacks is not None and callbacks['test'] is not None:
                callbacks['test']()

        train_losses.append(train_logs['loss'])
        val_losses.append(val_logs['loss'])
        test_losses.append(test_logs['loss'])

        plot_losses(train_losses, val_losses, test_losses, name)

    return best_model


def test(model, test_dataloader, criterion, metrics, weight=None, device=torch.device('cuda:0'), self_supervised=False, multiclass=False):
    test_logs, test_cm = run(
        model=model, dataloader=test_dataloader,
        criterion=criterion, weight=weight, optimizer=None,
        metrics=metrics, phase='test', device = device, self_supervised=self_supervised,
        multiclass=multiclass
    )

    print('TEST | ', end='')
    report_metrics(test_logs)
    return test_logs, test_cm
