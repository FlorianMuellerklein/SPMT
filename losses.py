from sys import maxsize

from requests import JSONDecodeError
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPMTLoss(nn.Module):
    def __init__(self, cfg, loss_proportion = 0.0):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.loss_proportion = loss_proportion
        self.iterations = 0.
        self.warmup_iterations = 5. * 89

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

            mask = torch.ones(n).long().to(pred[0].device)

            # if self paced learning we'll use attention over entropy to scale consistency loss
            # if self.cfg.spl:
            #     with torch.no_grad():
            #         # sort with entropy of temporal ensemble
            #         tempens_entorpy = torch.special.entr(torch.softmax(ema_logit, -1)).sum(-1)
            #         _, indices = torch.sort(tempens_entorpy)
            #         # choose the lowest k% of entropy
            #         chosen_indices = indices[:int(n * perc_take)]
            #         # apply hards attention over entropy
            #         mask[~chosen_indices] = 0

            unsupervised_loss = F.mse_loss(
                F.softmax(pred[1], -1),
                F.softmax(ema_logit, -1),
                size_average='mean'
            )

            max_lamb = 100.
            unsup_lambda = min(max_lamb, (float(self.iterations) / self.warmup_iterations) * max_lamb)
            self.iterations += 1

            unsupervised_loss = unsup_lambda * unsupervised_loss
            #unsupervised_loss = (unsupervised_loss.mean(-1) *  mask).sum() / mask.sum()
        else:
            unsupervised_loss = torch.tensor([0.]).to(pred[0].device)

        if self.cfg.jsd and aug_pred is not None:
            jsd_loss = 12. * self.jsd_loss(pred[0], aug_pred[0])
        else:
            jsd_loss = torch.tensor([0.]).to(pred[0].device)

        res_loss = 0.01 * self.symmetric_mse_loss(pred[0], pred[1])

        return supervised_loss, unsupervised_loss, res_loss, jsd_loss

    def update_sp_threshold(self, e: int):
        self.loss_proportion = min(1., e / (self.cfg.epochs / 4))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    def symmetric_mse_loss(self, input_a, input_b):
        assert input_a.size() == input_b.size()

        return torch.mean((input_a - input_b)**2)

    def jsd_loss(self, input_a, input_b):

        kl_targ = (F.softmax(input_a, -1) + F.softmax(input_b, -1)) / 2.

        kl_sum = (
            F.kl_div(F.log_softmax(input_a, -1), kl_targ ,reduction='batchmean') +
            F.kl_div(F.log_softmax(input_b, -1), kl_targ ,reduction='batchmean')
        )

        jsd_loss = 0.5 * (kl_sum)

        return jsd_loss
