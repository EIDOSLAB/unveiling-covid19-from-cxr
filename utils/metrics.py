import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class Metric:
    def __init__(self, multiclass=False):
        self.reset()
        self.multiclass = multiclass

    def reset(self):
        self.outputs = None
        self.targets = None
        self.phase = 'test'

    def accumulate(self, output, target, phase='test'):
        self.phase = phase

        if self.outputs is None:
            self.outputs = output.detach().cpu()
            self.targets = target.cpu()
        else:
            self.outputs = torch.cat((self.outputs, output.detach().cpu()))
            self.targets = torch.cat((self.targets, target.cpu()))

    def __repr__(self):
        return f'{self.get()}'

class Accuracy(Metric):
    __name__ = 'acc'

    def get(self, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()

        return accuracy_score(self.targets.numpy(), outputs.numpy())

    def get_best_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.targets.numpy(), self.outputs.numpy())
        best_idx = np.argmax(tpr-fpr)
        return thresholds[best_idx]

class RocAuc(Metric):
    __name__ = 'auc'

    def get(self):
        return roc_auc_score(self.targets.numpy(), self.outputs.numpy())

    def get_curve(self):
        return roc_curve(self.targets.numpy(), self.outputs.numpy())

class FScore(Metric):
    __name__ = 'f-score'

    def get(self, threshold=0.5):
        self.outputs = (self.outputs > threshold).long()
        cm = confusion_matrix(self.targets.numpy(), self.outputs.numpy())
        tn, fp, fn, tp = cm.ravel()
        recall = tp/(tp + fn)
        precision = tp/((tp + fp))
        return 2 * (recall * precision)/(precision + recall)

class ConfusionMatrix(Metric):
    __name__ = 'cm'

    def get(self, normalized=False, threshold=0.5):
        outputs = self.outputs
        if not self.multiclass:
            outputs = (outputs > threshold).long()
        cm = confusion_matrix(self.targets.numpy(), outputs.numpy())

        if normalized:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        return cm
