import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet
from runner import CNNTrainer
from losses import SPMTLoss
from data_loader import get_data_loaders

from warmup_scheduler import AddWarmup

parser = argparse.ArgumentParser(description='Semisupervised Training')
parser.add_argument(
    '--lr', default=0.05, type=float,
    help = 'learning rate'
)
parser.add_argument(
    '--warmup', default=0, type=int,
    help = 'number of epochs warmup learning rate'
)
parser.add_argument(
    '--batch_size', default=128, type=int,
    help = 'how large are mini-batches'
)
parser.add_argument(
    '--epochs', default=300, type=int,
    help = 'how large are mini-batches'
)
parser.add_argument(
    '--num_labeled', default=50000, type=int,
    help = 'how many labeled images to use from CIFAR10'
)
parser.add_argument('--spl', action='store_true',
    help = 'whether to use self paced learning'
)
parser.add_argument('--tempens', action='store_true',
    help = 'whether to use temporal ensembling'
)
parser.add_argument('--mt', action='store_true',
    help = 'whether to use mean teacher training'
)
parser.add_argument('--debug', action='store_true',
    help = 'whether to use soft pseudo labels training'
)
parser.add_argument('--model_name', type=str, default='MT',
    help = 'model name for saving experiment results'
)
args = parser.parse_args()

def main():
    # linearly scale learning rate with batch size
    args.lr = args.lr * (args.batch_size / 128)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # set up network
    net = ResNet(n=9)
    net = net.to(device)

    ema_net = ResNet(n=9)
    ema_net = ema_net.to(device)

    train_loader, valid_loader, test_loader = get_data_loaders(
        args.batch_size,
        args.num_labeled
    )

    # set up training loss and optimizer (using params from resnet paper)
    criterion = SPMTLoss(cfg = args) # nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.SGD(
        net.parameters(),
        lr = args.lr,
        momentum = 0.9,
        weight_decay = 0.0002,
        nesterov = True
    )

    # cosine learning rate decay with warmup
    scheduler = AddWarmup(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = (args.epochs + 25) * len(train_loader),
            eta_min = 0.,
            verbose = False
        ),
        warmup_dur = args.warmup * len(train_loader),
        starting_lr = args.lr * 0.0001,
        ending_lr = args.lr
    )

    # set up our trainer
    runner = CNNTrainer(
        train_loader = train_loader,
        valid_loader = valid_loader,
        test_loader = test_loader,
        student = net,
        teacher = ema_net,
        crit = criterion,
        device = device,
        optimizer = optimizer,
        scheduler = scheduler,
        cfg = args,
    )

    # train and test the network
    runner.train_network()

    training_type = 'mt' if args.mt else 'vanilla'
    training_type = training_type + '_tempens' if args.tempens else ''
    training_type = training_type + '_spl' if args.spl else ''

    df_acc = pd.DataFrame(runner.accuracies)
    df_loss = pd.DataFrame(runner.full_losses)
    df_unsup_loss = pd.DataFrame(runner.unsup_losses)

    df_acc.to_csv('results/{}_accuracy_curves_{}.csv'.format(args.model_name, training_type))
    df_loss.to_csv('results/{}_loss_curves_{}.csv'.format(args.model_name, training_type))
    df_unsup_loss.to_csv('results/{}_unsuploss_curves_{}.csv'.format(args.model_name, training_type))

    # get the test set performance
    test_err = runner.test_network()

    results = '{},{}'.format(
        training_type, test_err
    )

    with open('results/results.csv', 'a+') as f:
        f.write(results)
        f.write('\n')


if __name__ == '__main__':
    main()