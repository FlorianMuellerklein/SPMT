
from calendar import c
import sys
import time
import argparse

import numpy as np

import torch
import torchvision as vsn
from torchvision.utils import make_grid

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class MTTrainer:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader = None,
        valid_loader: torch.utils.data.DataLoader = None,
        test_loader: torch.utils.data.DataLoader = None,
        device: torch.device = None,
        student: torch.nn.Module = None,
        teacher: torch.nn.Module = None,
        crit: torch.nn.Module = None,
        cfg: argparse.Namespace = None,
        pseudo_crit: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    ):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.student = student
        self.teacher = teacher

        for param in self.teacher.parameters():
            param.detach_()

        self.crit = crit
        self.pseudo_crit = pseudo_crit
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        # initialize update step counter
        self.iterations = {'train': 0, 'valid': 0}
        self.unsup_iterations = 0

        # loss tracking
        self.full_losses = {'train': [], 'valid': []}
        self.accuracies = {'train': [], 'valid': []}
        self.ecr_losses = {'train': [], 'valid': []}
        self.cpl_losses = {'train': [], 'valid': []}
        self.pseudo_batch_sizes = []
        self.valid_err = []

        # rampup
        self.loss_rampup = np.linspace(0, 30, cfg.epochs // 2)


    def train_network(self):
        '''
        Train the network using the train and valid functions.

        Record the losses and saves model weights periodically

        Allows keyboard interrupt to stop training at any time
        '''
        try:
            print()
            print('-----------------------------------------')
            print('Training ...')
            print('-----------------------------------------')
            print()

            val_acc = 0.

            for e in range(self.cfg.epochs):

                print('\n' + 'Iter {}/{}'.format(e + 1, self.cfg.epochs))
                start = time.time()

                _ = self.run_epoch(mode = 'train', epoch = e)

                with torch.no_grad():
                    v_error = self.run_epoch(mode = 'valid', epoch = e)
                self.valid_err.append(1. - v_error)

                if self.cfg.debug and e > 1:
                    break

                print('Time: {}'.format(time.time() - start))

        except KeyboardInterrupt:
            pass

        self.update_bn()

        return min(self.valid_err)

    def run_epoch(self, mode: str, epoch: int) -> float:
        '''
        Uses the data loader to grab a batch of images

        Pushes images through network and gathers predictions

        Updates network weights by evaluating the loss functions
        '''

        running_loss = 0.
        running_total = 0
        running_corrects = 0

        running_ecr_loss = 0.
        running_cpl_loss = 0.

        if mode == 'train':
            # zero gradients
            self.optimizer.zero_grad(set_to_none = True)
            self.student.train(True)
            self.teacher.train()

            iterator = enumerate(self.train_loader)
            n_batches = len(self.train_loader)
        else:
            iterator = enumerate(self.valid_loader)
            n_batches = len(self.valid_loader)
            self.student.eval()
            self.teacher.eval()

        for i, data in iterator:
            imgs, targets = data['img'], data['targ']
            imgs, targets = imgs.to(self.device), targets.to(self.device)

            if mode == 'train' and (self.cfg.cpl or self.cfg.mr):
                ema_imgs = data['ema_img']
                ema_imgs = ema_imgs.to(self.device)

                with torch.no_grad():
                    ema_logit, features = self.teacher(ema_imgs)

            else:
                ema_logit, features = None, None

            # get predictions
            preds, _ = self.student(imgs)

            # calculate loss
            supervised_loss, cons_loss, pseudo_loss = self.crit(
                preds, targets, ema_logit, features, training_mode = mode=='train'
            )

            loss = supervised_loss + cons_loss + pseudo_loss

            if mode == 'train':

                # zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                loss.backward()

                # update weights
                self.optimizer.step()

                # count grad updates
                self.iterations[mode] += 1

                # step the LR scheduler
                self.scheduler.step()

                # update teacher
                self.update_teacher()

            # track statistics
            predicted = torch.argmax(preds, 1)
            mask = targets.ne(-1)
            total = max(mask.sum().item(), 1e-8)
            correct = (predicted[mask] == targets[mask]).sum().item()

            running_loss += supervised_loss.item()
            running_ecr_loss += cons_loss.item()
            running_cpl_loss += pseudo_loss.item()
            running_total += total
            running_corrects += correct


            # make a cool terminal output
            sys.stdout.write('\r')
            sys.stdout.write('{} B: {:>3}/{:<3} | Class: {:.3} | Cons: {:.3} | CPL: {:.3}'.format(
                mode,
                i+1,
                n_batches,
                supervised_loss.item(),
                cons_loss.item(),
                #res_loss.item(),
                pseudo_loss.item()
            ))
            sys.stdout.flush()

            if self.cfg.debug and i == 0:
                self.debug(data, mode, epoch)

                if epoch > 1:
                    break

        print(
            '\n' + 'Avg Class: {:.4} | Acc: {:.2} | ECR Loss: {:.4} | CPL Loss: {:.4}'.format(
                running_loss / n_batches,
                running_corrects / running_total,
                running_ecr_loss / n_batches,
                running_cpl_loss / n_batches
            )
        )

        self.full_losses[mode].append(running_loss / n_batches)
        self.ecr_losses[mode].append(running_ecr_loss / n_batches)
        self.cpl_losses[mode].append(running_cpl_loss / n_batches)
        self.accuracies[mode].append(running_corrects / (running_total + 1e-8))

        return running_corrects / (running_total + 1e-8)


    def make_tsne(self, name) -> float:

        if self.cfg.mr:
            eval_net = self.teacher.eval()
        else:
            eval_net = self.student.eval()

        testset_feats = []
        testset_targs = []

        with torch.no_grad():
            for data in self.valid_loader:
                imgs, targets = data
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                # get predictions
                _, features = eval_net(imgs)

                testset_feats.append(features.cpu().numpy())
                testset_targs.extend(targets.cpu().numpy().tolist())

        testset_feats = np.concatenate(testset_feats)

        tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(testset_feats)

        plt.scatter(tsne[:,0], tsne[:,1], c=np.array(testset_targs), edgecolors='black')
        plt.xticks([])
        plt.yticks([])
        plt.grid(linestyle=':')
        plt.savefig('imgs/tsne_{}.png'.format(name))

    def update_bn(self,):
        self.teacher.train()

        for data in self.train_loader:
            _, _ = self.teacher(data['img'].to(self.device))

    def update_teacher(self):
        # use regular average until model weights stabilize
        alpha = min(1. - 1 / (self.iterations['train'] + 1), self.cfg.alpha)

        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data *= alpha                       # previous data multiplied by weight
            param_t.data += param_s.data * (1. - alpha) # new data multiplied by inv weight


    def debug(self, data, mode, epoch):
        # for debugging that dataloader is working properly
        grid = vsn.utils.make_grid(data['img'])
        vsn.utils.save_image(grid, f'imgs/{mode}_{epoch}_imgs.png')

        if mode == 'train':
            grid = vsn.utils.make_grid(data['img_consistency'])
            vsn.utils.save_image(grid, f'imgs/{mode}_{epoch}_imgs-consist.png')

            grid = vsn.utils.make_grid(data['unlabeled_img_a'])
            vsn.utils.save_image(grid, f'imgs/{mode}_{epoch}_imgs-unlab-a.png')

            grid = vsn.utils.make_grid(data['unlabeled_img_b'])
            vsn.utils.save_image(grid, f'imgs/{mode}_{epoch}_imgs-unlab-b.png')
