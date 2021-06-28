import torch
from ignite.metrics import Metric
from sklearn.metrics import f1_score


class F1Score(Metric):

    def __init__(self, *args, **kwargs):
        self.f1 = 0
        self.count = 0
        super().__init__(*args, **kwargs)

    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        _, predicted = torch.max(y_pred, 1)
        f = f1_score(y.cpu(), predicted.cpu(), average='micro')
        self.f1 += f
        self.count += 1

    def reset(self):
        self.f1 = 0
        self.count = 0
        super(F1Score, self).reset()

    def compute(self):
        return self.f1 / self.count
