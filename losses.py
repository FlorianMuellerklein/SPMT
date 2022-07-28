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

            mr_lambda: float = 100.,
            cpl_lambda: float = 0.5,
            label_smoothing: float = 0.1,

        ):
        super(SPMTLoss, self).__init__()

        self.cfg = cfg
        self.mr_lambda = mr_lambda
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
            student_logits: Tensor,
            targ_class: Tensor,
            teacher_logits: Tensor,
            features: Tensor,
            training_mode: bool = True
        ):

        supervised_loss = torch.tensor([0.]).to(student_logits.device)
        cons_loss = torch.tensor([0.]).to(student_logits.device)
        pseudo_loss = torch.tensor([0.]).to(student_logits.device)

        # normal supervsied cross entropy
        supervised_loss += self.class_crit(student_logits, targ_class)

        if self.cfg.mr and (teacher_logits is not None) and (features is not None):
            # manifold regularization
            cons_loss += self.manifold_regularization(student_logits, teacher_logits, features)

        if self.cfg.cpl:
            # curriculum pseudo loss
            pseudo_loss += self.curriculum_pseudo_loss(student_logits, teacher_logits, targ_class)

        if training_mode:
            self.iterations += 1

        return supervised_loss, cons_loss, pseudo_loss

    def curriculum_pseudo_loss(self, student_logits, teacher_logits, targ_class):
        '''
        Curriculum Pseudo Labels similar to https://arxiv.org/pdf/2109.00650v1.pdf
        '''
        # mask only the unlabeled targets
        unlabeled_mask = targ_class.eq(-1)

        # use an offset for the current iteration
        offset_cur_iter = max(0., (self.iterations - self.cpl_iter_offset))

        pseudo_loss = 0.

        if teacher_logits is not None:

            # find the percentile mask
            with torch.no_grad():
                confidences, pseudo_targets = torch.softmax(teacher_logits[unlabeled_mask], -1).max(-1)
                confidence_threshold = self.get_percentile(confidences, offset_cur_iter)
                mask = confidences >= confidence_threshold

            if mask.sum() > 0:

                pseudo_loss = F.cross_entropy(
                    student_logits[unlabeled_mask][mask],
                    pseudo_targets[mask],
                )

        cpl_rampup = self.rampup(offset_cur_iter, self.cpl_warmup_iterations)

        return cpl_rampup * self.cpl_lambda * pseudo_loss


    def manifold_regularization(self, student_logits, teacher_logits, features):
        '''
        Manifold Regularization enforcing consistency between predictions for the k-nearest points in
        the learned embedding space as given by cosine similarity.

        Setting k to 1 recovers ECR from https://arxiv.org/abs/1703.01780 and https://arxiv.org/abs/2109.14563
        '''
        # pairwise cosine similarity
        #sims = features / torch.linalg.norm(features, dim=-1).unsqueeze(1)
        #sims = sims @ sims.T

        # gaussian kernel
        sims = torch.linalg.norm(features.unsqueeze(1) - features[:, :] + 1e-8, dim=-1)
        sims = 1 / (1 + sims)

        # pairwise mse
        mse = ((F.softmax(student_logits, -1).unsqueeze(1) - F.softmax(teacher_logits, -1)) ** 2).mean(-1)

        # get the k nearest
        _, k_nearest = torch.topk(sims, self.cfg.knn, dim=-1)
        # average cosine similarity weighted mse on the k nearest points for each member of batch
        cons_loss = torch.mean(sims.gather(1, k_nearest) * mse.gather(1, k_nearest))

        cons_loss = self.mr_lambda * self.rampup(self.iterations, self.ecr_warmup_iterations) * cons_loss

        return cons_loss

    def rampup(self, cur_iter, warmup_iterations):
        '''Linear rampup function for loss scaling lambdas'''
        return min(1., (cur_iter / warmup_iterations))

    def entropy(self, dist):
        return -1 * torch.sum((dist + 1e-8) * torch.log(dist + 1e-8), axis=-1)

    def get_percentile(self, confidences, cur_iter):
        # add percentials by 10% at a time
        ratio = math.ceil((cur_iter / self.total_iterations) * 10) *  10.
        quantile = max(0., 1. - (ratio / 100.))

        return torch.quantile(confidences, quantile, interpolation='linear')