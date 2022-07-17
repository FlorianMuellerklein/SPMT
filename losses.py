import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPMTLoss(nn.Module):
    def __init__(
            self,
            cfg,
            ecr_warmup_iterations: int,
            cpl_warmup_iterations: int,
            cpl_iter_offset: int,
            total_iterations: int,

            cons_lambda: float = 100.,
            cpl_lambda: float = 1.,
            label_smoothing: float = 0.1,

        ):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.gamma = 3.
        self.cons_lambda = cons_lambda
        self.cpl_lambda = cpl_lambda
        self.label_smoothing = label_smoothing
        self.cpl_iter_offset: int = cpl_iter_offset

        self.iterations = 0.
        self.ecr_warmup_iterations = ecr_warmup_iterations
        self.cpl_warmup_iterations = cpl_warmup_iterations
        self.total_iterations = total_iterations

        # if self.cfg.spl:
        #self.class_crit = self.softmax_focal
        # else:
        self.class_crit = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=-1,
            label_smoothing = label_smoothing
        )

    def forward(
        self,
        pred: Tensor,
        targ_class: Tensor,
        ema_logit: Tensor,
        training_mode: bool = True
    ):

        supervised_loss = torch.tensor([0.]).to(pred.device)
        cons_loss = torch.tensor([0.]).to(pred.device)
        pseudo_loss = torch.tensor([0.]).to(pred.device)

        # normal supervsied cross entropy
        supervised_loss += self.class_crit(pred, targ_class)

        if self.cfg.ecr:
            # consistency ECR loss from mean teacher
            cons_loss += self.consistency_loss(pred, ema_logit)

        if self.cfg.cpl:
            # curriculum pseudo loss
            pseudo_loss += self.curriculum_pseudo_loss(pred, ema_logit, targ_class)

        if training_mode:
            self.iterations += 1

        return supervised_loss, cons_loss, pseudo_loss

    def softmax_focal(self, input, target):

        mask = target != -1
        input = input[mask]
        target = target[mask]

        range_n = torch.arange(0, target.size(0), dtype=torch.int64, device=target.device)

        logpt = F.log_softmax(input, 1)
        logpt = logpt[range_n, target.long()]

        # with torch.no_grad():
        #     smoothed_targets = torch.zeros_like(logpt)
        #     smoothed_targets.fill_(self.label_smoothing / (input.size(1) - 1))
        #     smoothed_targets[range_n, target] = 1. - self.label_smoothing

        pt = logpt.exp()

        loss = -1 * (1. - pt) ** self.gamma * logpt

        return loss.mean()

    def curriculum_pseudo_loss(self, pred, ema_logit, targ_class):
        '''
        Curriculum Pseudo Labels from https://arxiv.org/pdf/2109.00650v1.pdf
        '''
        # mask only the unlabeled targets
        unlabeled_mask = targ_class.eq(-1)

        # use an offset for the current iteration
        offset_cur_iter = max(0., (self.iterations - self.cpl_iter_offset))

        pseudo_loss = 0.

        if ema_logit is not None:

            # find the percentile mask
            with torch.no_grad():
                confidences, pseudo_targets = torch.softmax(ema_logit[unlabeled_mask], -1).max(-1)
                confidence_threshold = self.get_percentile(confidences, offset_cur_iter)
                mask = confidences >= confidence_threshold

            if mask.sum() > 0:

                pseudo_loss = F.cross_entropy(
                    pred[unlabeled_mask][mask],
                    pseudo_targets[mask],
                    label_smoothing = self.label_smoothing
                )
                # pseudo_loss = self.softmax_focal(
                #     pred[unlabeled_mask][mask],
                #     pseudo_targets[mask]
                # )

        cpl_rampup = self.rampup(offset_cur_iter, self.cpl_warmup_iterations)

        return cpl_rampup * self.cpl_lambda * pseudo_loss

    def consistency_loss(self, pred, ema_logit):
        '''
        Consistency loss component consisting of ECR similar to
        https://arxiv.org/abs/1703.01780 and https://arxiv.org/abs/2109.14563
        '''
        if (ema_logit is not None):

            # calculate the unsupervised loss for the labeled portion
            cons_loss = F.mse_loss(
                F.softmax(pred, -1),
                F.softmax(ema_logit, -1),
                reduction='mean'
            )

            cons_loss = self.cons_lambda * self.rampup(self.iterations, self.ecr_warmup_iterations) * cons_loss
        else:
            cons_loss = torch.tensor([0.]).to(pred.device)

        return cons_loss

    def manifold_regularization(self, pred, ema_logit, features):

        if (ema_logit is not None):

            # calculate all pairwise distances
            dists = torch.linalg.norm(features[:, None, :] - features[:, :] + 1e-8, dim=-1)
            # invert all the values. Closest euc distance has maximum weight
            dists = dists.max() - dists

            # pairwise mse
            mse = ((F.softmax(pred, -1)[:, None, :] - F.softmax(ema_logit, -1)) ** 2).mean(-1)

            # average euclidean distance weighted mse on the lower triangle
            cons_loss = torch.mean(torch.tril(dists, diagonal=-1) * torch.tril(mse, diagonal=-1))

            cons_loss = self.cons_lambda * self.rampup(self.iterations, self.ecr_warmup_iterations) * cons_loss
        else:
            cons_loss = torch.tensor([0.]).to(pred.device)

        return cons_loss

    def rampup(self, cur_iter, warmup_iterations):
        '''Linear rampup function for loss scaling lambdas'''
        return min(1., (cur_iter / warmup_iterations))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    # def symmetric_mse_loss(self, input_a, input_b):
    #     '''
    #     Symmetric version of MSE that send grads to both inputs. https://arxiv.org/abs/1703.01780
    #     '''
    #     assert input_a.size() == input_b.size()

    #     return torch.mean((input_a - input_b)**2)

    # def jsd_loss(self, input_a, input_b):
    #     '''
    #     Jensen-Shannon Divergence loss implemented similarly to: https://arxiv.org/abs/2109.14563
    #     '''
    #     assert input_a.size() == input_b.size()

    #     kl_targ = 0.5 * (F.softmax(input_a, -1) + F.softmax(input_b, -1))

    #     jsd_loss = (
    #         0.5 * F.kl_div(F.log_softmax(input_a, -1), kl_targ.detach(), reduction='batchmean') +
    #         0.5 * F.kl_div(F.log_softmax(input_b, -1), kl_targ.detach(), reduction='batchmean')
    #     )

    #     return jsd_loss

    def get_percentile(self, confidences, cur_iter):
        # add percentials by 10% at a time
        ratio = math.ceil((cur_iter / self.total_iterations) * 10) *  10.
        quantile = max(0., 1. - (ratio / 100.))

        return torch.quantile(confidences, quantile, interpolation='linear')
