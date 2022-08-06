import argparse
from copy import deepcopy

import pandas as pd

import torch
import torch.optim as optim

from model import ResNet
from runner import MTTrainer
from losses import SPMTLoss
from data_loader import get_data_loaders

from warmup_scheduler import AddWarmup

parser = argparse.ArgumentParser(description='Semisupervised Training')
parser.add_argument(
    '--lr', default=0.1, type=float,
    help = 'learning rate'
)
parser.add_argument(
    '--wd', default=2e-4, type=float,
    help = 'weight decay'
)
parser.add_argument(
    '--alpha', default=0.99, type=float,
    help = 'alpha for exponential moving average'
)
parser.add_argument(
    '--mr_lambda', default=100., type=float,
    help = 'scaling factor on manifold regularization'
)
parser.add_argument(
    '--cpl_lambda', default=.75, type=float,
    help = 'scaling factor on curriculum pseudo labels'
)
parser.add_argument(
    '--warmup', default=1, type=int,
    help = 'number of epochs warmup learning rate'
)
parser.add_argument(
    '--batch_size', default=256, type=int,
    help = 'how large are mini-batches'
)
parser.add_argument(
    '--epochs', default=225, type=int,
    help = 'how large are mini-batches'
)
parser.add_argument(
    '--num_labeled', default=50000, type=int,
    help = 'how many labeled images to use from CIFAR10'
)
parser.add_argument(
    '--spl', action='store_true',
    help = 'whether to use self paced learning'
)
parser.add_argument(
    '--mr', action='store_true',
    help = 'whether to use manifold regularization'
)
parser.add_argument(
    '--knn', default=1, type=int,
    help = 'how many neighbors to use in manifold regularization'
)
parser.add_argument(
    '--cpl', action='store_true',
    help = 'whether to use curriculum pseudo labeling'
)
parser.add_argument(
    '--debug', action='store_true',
    help = 'whether to use soft pseudo labels training'
)
parser.add_argument(
    '--model_name', type=str, default='ResNet56',
    help = 'model name for saving experiment results'
)
args = parser.parse_args()

def main():
    # linearly scale learning rate with batch size
    args.lr = args.lr * (args.batch_size / 128)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up networks
    net = ResNet(n=9)
    ema_net = deepcopy(net)

    net = net.to(device)
    ema_net = ema_net.to(device)

    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    ema_net = torch.nn.DataParallel(ema_net, device_ids=[0, 1])

    train_loader, valid_loader = get_data_loaders(
        args.batch_size,
        args.num_labeled
    )

    # set up training loss and optimizer (using params from resnet paper)
    criterion = SPMTLoss(
        cfg = args,
        mr_lambda = args.mr_lambda,
        ecr_warmup_iterations = 5. * len(train_loader),  # warmup ensemble consistency for 5 epochs
        cpl_warmup_iterations = 50. * len(train_loader), # warmup curriculum pseudos for 50 epochs
        cpl_iter_offset = 50. * len(train_loader),       # start cpl loss after 50 epochs
        total_iterations = args.epochs * len(train_loader),
    )

    optimizer = optim.SGD(
        net.parameters(),
        lr = args.lr,
        momentum = 0.9,
        weight_decay = args.wd,
        nesterov = True
    )

    # cosine learning rate decay with warmup
    scheduler = AddWarmup(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = (args.epochs - args.warmup + 10) * len(train_loader),
            eta_min = 0.,
            verbose = False
        ),
        warmup_dur = args.warmup * len(train_loader),
        starting_lr = args.lr * 0.0001,
        ending_lr = args.lr
    )

    # set up our trainer
    runner = MTTrainer(
        train_loader = train_loader,
        valid_loader = valid_loader,
        student = net,
        teacher = ema_net,
        crit = criterion,
        device = device,
        optimizer = optimizer,
        scheduler = scheduler,
        cfg = args,
    )

    # train and test the network
    test_err = runner.train_network()
    runner.make_tsne()

    training_type = 'mr' + '_k-{}_'.format(args.knn) if args.mr else 'vanilla'
    training_type += training_type + '_cpl' if args.cpl else ''

    df_acc = pd.DataFrame(runner.accuracies)
    df_loss = pd.DataFrame(runner.full_losses)
    df_ecr_loss = pd.DataFrame(runner.ecr_losses)
    df_cpl_loss = pd.DataFrame(runner.cpl_losses)

    df_acc.to_csv('results/{}_accuracy_curves_{}.csv'.format(args.model_name, training_type))
    df_loss.to_csv('results/{}_loss_curves_{}.csv'.format(args.model_name, training_type))
    df_ecr_loss.to_csv('results/{}_ecr_curves_{}.csv'.format(args.model_name, training_type))
    df_cpl_loss.to_csv('results/{}_cpl_curves_{}.csv'.format(args.model_name, training_type))

    results = '{},{}'.format(
        args.model_name + '_' + training_type, test_err
    )

    with open('results/results.csv', 'a+') as f:
        f.write(results)
        f.write('\n')


if __name__ == '__main__':
    main()