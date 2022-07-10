
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPMTLoss(nn.Module):
    def __init__(self, cfg, warmup_iterations, total_iterations):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.iterations = 0.
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations

        self.class_crit = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1, label_smoothing = 0.1)

    def forward(
        self,
        pred: Tensor,
        targ_class: Tensor,
        ema_logit: Tensor,
        aug_pred: Tensor
    ):

        supervised_loss = 0. + self.class_crit(pred, targ_class)

        unsupervised_loss = self.unsupervised_loss(pred, ema_logit, targ_class)

        if self.cfg.jsd and aug_pred is not None:
            jsd_loss = 10. * self.rampup() * self.jsd_loss(pred, aug_pred)
        else:
            jsd_loss = torch.tensor([0.]).to(pred.device)

        self.iterations += 1

        return supervised_loss, unsupervised_loss, jsd_loss

    def rampup(self):
        '''Linear rampup function for loss scaling lambdas'''
        return min(1., (float(self.iterations) / self.warmup_iterations))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    def unsupervised_loss(self, pred, ema_logit, targ_class):
        '''
        Unsupervised loss component consisting of ECR similar to
        https://arxiv.org/abs/1703.01780 and https://arxiv.org/abs/2109.14563

        Self paced learning strategy from https://arxiv.org/pdf/2109.00650v1.pdf
        '''
        if (ema_logit is not None):

            labeled_mask = targ_class.ne(-1)

            # calculate the unsupervised loss for the labeled portion
            labeled_loss = F.mse_loss(
                F.softmax(pred[labeled_mask], -1),
                F.softmax(ema_logit[labeled_mask], -1),
                reduction='mean'
            )

            # unsupervised_loss = F.cross_entropy(
            #     pred[1],
            #     torch.argmax(ema_logit, -1),
            #     size_average='none',
            #     label_smoothing = 0.1
            # )

            if self.cfg.spl:
                #entropy = torch.special.entr(F.softmax(ema_logit,-1)).sum(-1)
                #attn = F.softmax(-entropy, -1)
                #unsupervised_loss = (unsupervised_loss.mean(-1) * attn).sum()

                # using the strategy from Curriculum Pseudo Labels: https://arxiv.org/pdf/2109.00650v1.pdf

                if labeled_mask.sum() > 0:
                    unlabeled_loss = 0. + F.mse_loss(
                            F.softmax(pred[~labeled_mask], -1),
                            F.softmax(ema_logit[~labeled_mask], -1),
                            reduction='none'
                        ).mean(-1)

                    # only keep unsupervised loss below this threshold
                    threshold = self.get_percentile(unlabeled_loss)
                    unsup_mask = unlabeled_loss < threshold

                    unlabeled_loss = (unlabeled_loss * unsup_mask.detach()).sum() / unsup_mask.sum()
                else:
                    unlabeled_loss = 0.

            else:
                # use all the unlabeled data
                unlabeled_loss = F.mse_loss(
                        F.softmax(pred[~labeled_mask], -1),
                        F.softmax(ema_logit[~labeled_mask], -1),
                        reduction='mean'
                    )

            unsup_lambda = 100. * self.rampup()
            unsupervised_loss = unsup_lambda * (unlabeled_loss + labeled_loss)
        else:
            unsupervised_loss = torch.tensor([0.]).to(pred.device)

        return unsupervised_loss

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

    def get_percentile(self, losses):
        quantile = (self.iterations + 1) / self.total_iterations
        return torch.quantile(losses, quantile, interpolation='linear')
