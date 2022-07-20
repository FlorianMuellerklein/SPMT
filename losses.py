import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity

class SPMTLoss(nn.Module):
    def __init__(
            self,
            cfg,
            ecr_warmup_iterations: int,
            cpl_warmup_iterations: int,
            cpl_iter_offset: int,
            total_iterations: int,

            cons_lambda: float = 100.,
            cpl_lambda: float = 0.5,
            gamma: float = 50.,
            k: int = 3,
            label_smoothing: float = 0.1,

        ):
        super(SPMTLoss, self).__init__()
        self.cfg = cfg
        self.k = k
        self.gamma = gamma
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

        self.triplet_loss = TipletMiningLoss()

    def forward(
        self,
        pred: Tensor,
        targ_class: Tensor,
        ema_logit: Tensor,
        features: Tensor,
        training_mode: bool = True
    ):

        supervised_loss = torch.tensor([0.]).to(pred.device)
        cons_loss = torch.tensor([0.]).to(pred.device)
        pseudo_loss = torch.tensor([0.]).to(pred.device)
        #triplet_loss = torch.tensor([0.]).to(pred.device)

        # normal supervsied cross entropy
        supervised_loss += self.class_crit(pred, targ_class)

        if self.cfg.ecr:
            # consistency ECR loss from mean teacher
            cons_loss += self.consistency_loss(pred, ema_logit)
        elif self.cfg.mr and (ema_logit is not None) and (features is not None):
            cons_loss += self.manifold_regularization(pred, ema_logit, features)
            #pseudo_loss += 0.0001 * self.triplet_loss(features, torch.argmax(ema_logit, -1))

        if self.cfg.cpl:
            # curriculum pseudo loss
            pseudo_loss += self.curriculum_pseudo_loss(pred, ema_logit, targ_class)

        if training_mode:
            self.iterations += 1

        return supervised_loss, cons_loss, pseudo_loss

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

        # calculate the unsupervised loss for the labeled portion
        cons_loss = F.mse_loss(
            F.softmax(pred, -1),
            F.softmax(ema_logit, -1),
            reduction='mean'
        )

        cons_loss = self.cons_lambda * self.rampup(self.iterations, self.ecr_warmup_iterations) * cons_loss

        return cons_loss

    def manifold_regularization(self, pred, ema_logit, features):

        # calculate all pairwise distances, functions as a graph adjacency matrix for all points in batch
        #dists = torch.linalg.norm(features[:, None, :] - features[:, :] + 1e-8, dim=-1)
        # normalize then invert all the values. Closest euc distance has maximum weight of 1
        #dists = (dists - dists.min()) / (dists.max() - dists.min())
        #dists = 1. - dists
        #dists = dists ** self.gamma

        # cosine distances
        dists = torch.tensor(cosine_similarity(features.cpu().numpy())).to(pred.device)

        # pairwise mse
        mse = ((F.softmax(pred, -1)[:, None, :] - F.softmax(ema_logit, -1)) ** 2).mean(-1)

        # get the k nearest
        _, k_nearest = torch.topk(dists, self.k, dim=-1)
        # average euclidean distance weighted mse on the k nearest points for each member of batch
        cons_loss = torch.mean(dists.gather(1, k_nearest) * mse.gather(1, k_nearest))

        cons_loss = self.cons_lambda * self.rampup(self.iterations, self.ecr_warmup_iterations) * cons_loss

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


class TipletMiningLoss(nn.Module):
    '''
    Online Triplet Loss with hard mining. Takes the same input as if one was using the normal
    CrossEntropyLoss with PyTorch. The inputs are a batch of embeddings (N, embed_dim) and a vector (N)
    of corresponding labels. The forward function finds for each embedding the furthest positive and
    closest negative and calculates the triplet loss using those.

    Parameters
    ----------
        margin: float
            The margin used for comparison between the pos and neg distances.

    Methods
    -------
        forward: vector_batch (N, embed_dim), labels_batch (N)
            Calculates the triplet loss and finds the furthest positives and closest negatives.

        get_positive_mask: labels (N)
            Finds the mask that contains valid positive matches. In other words it creates a binary
            mask of all inputs that have a matching label but ignores indices along the diagonal
            because that would be the same index as the current query vector.

        get_negative_mask: labels (N)
            Finds a mask that contains valid negative matches. In other words it creates a binary
            mask where two labels do not match.
    '''
    def __init__(self, margin = 1.):
        super(TipletMiningLoss, self).__init__()
        self.margin = margin

    def forward(self, vector_batch: torch.Tensor, labels_batch: torch.Tensor) -> torch.Tensor:

        # calculate all pairwise distances
        dists = torch.linalg.norm(vector_batch[:, None, :] - vector_batch[:, :] + 1e-8, dim=-1)

        # get the masks for valid positive matches and valid negative matches
        pos_mask = self.get_positive_mask(labels_batch).to(vector_batch.device)
        neg_mask = self.get_negative_mask(labels_batch).to(vector_batch.device)

        positive_dists = torch.max(dists * pos_mask, 1)[0].to(vector_batch.device)

        # min masking doesn't work because we'll just take 0 when the neg_mask is applied
        # so we take the maximum distance value and set that in the inverse of the negative mask
        # then when we take the min we won't accidentialy take a value where two indices match
        global_max_value = torch.max(dists).item()
        negative_dists = torch.min(dists + (global_max_value * ~neg_mask), 1)[0].to(vector_batch.device)

        # calculate triplet loss using mined pairs
        tl = torch.max(positive_dists - negative_dists + self.margin, torch.Tensor([0.0]).to(vector_batch.device))

        return torch.mean(tl)

    def get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        # ones everywhere except for diagonal (same index)
        diag_mask = torch.eye(labels.size(0)).bool().to(labels.device)
        diag_mask = ~diag_mask

        # same label
        equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

        # get the union of matching index and the diagonal mask
        mask = diag_mask & equal_mask

        return mask

    def get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        # get the makes for where labels don't match
        return torch.ne(labels.unsqueeze(0), labels.unsqueeze(1))
