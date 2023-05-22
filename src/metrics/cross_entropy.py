import torch.nn as nn
from torch.nn import CrossEntropyLoss

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = CrossEntropyLoss()

    @property
    def name(self):
        return "cross_entropy"

    @property
    def mode(self):
        return "min"

    def forward(self, pred, gt):
        return self.metric(pred, gt)