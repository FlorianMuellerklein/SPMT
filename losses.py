from sys import maxsize

from requests import JSONDecodeError
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPMTLoss(nn.Module):
    def __init__(self, cfg, warmup_iterations):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.iterations = 0.
        self.warmup_iterations = warmup_iterations

        self.class_crit = nn.CrossEntropyLoss(size_average='mean', ignore_index=-1)

    def forward(
        self,
        pred: Tensor,
        targ_class: Tensor,
        ema_logit: Tensor,
        aug_pred: Tensor
    ):
        n = pred[0].size(0)
        n_class = pred[0].size(1)

        supervised_loss = self.class_crit(pred[0], targ_class)

        if (ema_logit is not None):

            unsupervised_loss = F.mse_loss(
                F.softmax(pred[1], -1),
                F.softmax(ema_logit, -1),
                size_average='mean'
            )

            with torch.no_grad():
                unsup_lambda = supervised_loss.item() / (unsupervised_loss.item() + 1e-8)

            unsup_lambda = unsup_lambda * self.rampup()
            unsupervised_loss = unsup_lambda * unsupervised_loss
        else:
            unsupervised_loss = torch.tensor([0.]).to(pred[0].device)

        if self.cfg.jsd and aug_pred is not None:
            jsd_loss = self.rampup() * self.jsd_loss(pred[0], aug_pred[0])
        else:
            jsd_loss = torch.tensor([0.]).to(pred[0].device)

        res_loss = 0.01 * self.symmetric_mse_loss(pred[0], pred[1])

        self.iterations += 1

        return supervised_loss, unsupervised_loss, res_loss, jsd_loss

    def rampup(self):
        return min(1., (float(self.iterations) / self.warmup_iterations))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    def symmetric_mse_loss(self, input_a, input_b):
        assert input_a.size() == input_b.size()

        return torch.mean((input_a - input_b)**2)

    def jsd_loss(self, input_a, input_b):

        kl_targ = (F.softmax(input_a, -1) + F.softmax(input_b, -1)) / 2.

        kl_sum = (
            F.kl_div(F.log_softmax(input_a, -1), kl_targ.detach(), reduction='batchmean') +
            F.kl_div(F.log_softmax(input_b, -1), kl_targ.detach(), reduction='batchmean')
        )

        jsd_loss = 0.5 * (kl_sum)

        return jsd_loss
