
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPMTLoss(nn.Module):
    def __init__(self, cfg, ecr_warmup_iterations, cpl_warmup_iterations, total_iterations, cons_lambda = 100.):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.cons_lambda = cons_lambda

        self.iterations = 0.
        self.ecr_warmup_iterations = ecr_warmup_iterations
        self.cpl_warmup_iterations = cpl_warmup_iterations
        self.total_iterations = total_iterations

        self.class_crit = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=-1,
            label_smoothing = 0.1
        )

    def forward(
        self,
        pred: Tensor,
        targ_class: Tensor,
        ema_logit: Tensor,
    ):

        supervised_loss, cons_loss, pseudo_loss = torch.tensor([0.]).to(pred.device), torch.tensor([0.]).to(pred.device), torch.tensor([0.]).to(pred.device)

        # normal supervsied cross entropy
        supervised_loss += self.class_crit(pred, targ_class)

        if self.cfg.mt:
            # consistency ECR loss from mean teacher
            cons_loss += self.consistency_loss(pred, ema_logit)

        if self.cfg.spl:
            # curriculum pseudo loss
            pseudo_loss += self.curriculum_pseudo_loss(pred, ema_logit, targ_class)

        self.iterations += 1

        return supervised_loss, cons_loss, pseudo_loss


    def curriculum_pseudo_loss(self, pred, ema_logit, targ_class):
        '''
        Curriculum Pseudo Labels from https://arxiv.org/pdf/2109.00650v1.pdf
        '''
        # mask only the unlabeled targets
        unlabeled_mask = targ_class.eq(-1)

        if ema_logit is not None:
            with torch.no_grad():
                confidences, _ = torch.softmax(ema_logit[unlabeled_mask], -1).max(-1)

                confidence_threshold = self.get_percentile(confidences)

                mask = confidences >= confidence_threshold

                pseudo_loss = F.cross_entropy(
                    pred[unlabeled_mask][mask],
                    torch.argmax(ema_logit[unlabeled_mask][mask], -1),
                    label_smoothing=0.1
                )
        else:
            pseudo_loss = 0.

        return self.rampup(self.cpl_warmup_iterations) * pseudo_loss

    def consistency_loss(self, pred, ema_logit):
        '''
        Consistency loss component consisting of ECR similar to
        https://arxiv.org/abs/1703.01780 and https://arxiv.org/abs/2109.14563

        Self paced learning strategy from https://arxiv.org/pdf/2109.00650v1.pdf
        '''
        if (ema_logit is not None):


            # calculate the unsupervised loss for the labeled portion
            cons_loss = F.mse_loss(
                F.softmax(pred, -1),
                F.softmax(ema_logit, -1),
                reduction='mean'
            )


            cons_loss = self.cons_lambda * self.rampup(self.ecr_warmup_iterations) * cons_loss
        else:
            cons_loss = torch.tensor([0.]).to(pred.device)

        return cons_loss

    def rampup(self, warmup_iterations):
        '''Linear rampup function for loss scaling lambdas'''
        return min(1., (float(self.iterations) / warmup_iterations))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    def symmetric_mse_loss(self, input_a, input_b):
        '''
        Symmetric version of MSE that send grads to both inputs. https://arxiv.org/abs/1703.01780
        '''
        assert input_a.size() == input_b.size()

        return torch.mean((input_a - input_b)**2)

    def jsd_loss(self, input_a, input_b):
        '''
        Jensen-Shannon Divergence loss implemented similarly to: https://arxiv.org/abs/2109.14563
        '''
        assert input_a.size() == input_b.size()

        kl_targ = 0.5 * (F.softmax(input_a, -1) + F.softmax(input_b, -1))

        jsd_loss = (
            0.5 * F.kl_div(F.log_softmax(input_a, -1), kl_targ.detach(), reduction='batchmean') +
            0.5 * F.kl_div(F.log_softmax(input_b, -1), kl_targ.detach(), reduction='batchmean')
        )

        return jsd_loss

    def get_percentile(self, confidences):
        quantile = max(0., 1. - ((self.iterations + 1) / self.total_iterations))
        return torch.quantile(confidences, quantile, interpolation='linear')
