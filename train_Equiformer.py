import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np

import os
from pathlib import Path

from contextlib import suppress
from timm.utils import NativeScaler

from optim_factory import create_optimizer
from logger import FileLogger

from engine import train_one_step, evaluate, compute_stats
from torch.optim.lr_scheduler import LambdaLR
from common.adapters import equiformer_adapter
from common.data import build_dataloaders, load_dataset_splits
from common.training_utils import (
    build_spectrum_targets,
    save_pred,
    warmup_exponential_decay,
)

# distributed training
import utils as utils
from spectrum.write import save_spectrum

class OneBatchLoader:
    def __init__(self, batch): self.batch = batch
    def __iter__(self):
        yield self.batch
    def __len__(self):
        return 1

def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2_e3')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--num-basis', type=int, default=128)
    # training hyper-parameters
    parser.add_argument("--batch-size", type=int, default=64)
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--lr', type=float, default=5.0e-4, metavar='LR',
                        help='learning rate (default: 5.0e-4)')
    parser.add_argument('--min-lr', type=float, default=1.0e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1.0e-6)')

    parser.add_argument('--decay-step', type=int, default=10000, metavar='N',
                        help='step interval to decay LR')
    parser.add_argument('--lr-warmup-steps', type=int, default=1000, metavar='N',
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--lr-warmup-factor', type=float, default=0.1)
    parser.add_argument('--decay-rate', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task
    parser.add_argument("--data-path", type=str, default='IrDB')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='MAE')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--split-index-npz', type=str, default='IrDB/raw/splits.npz')

    parser.add_argument('--train-steps', type=int, default=10000)
    parser.add_argument('--eval-steps', type=int, default=100)
    
    #spectrum_calculation
    parser.add_argument('--spectrum-type', type=str, default='FC')
    parser.add_argument('--n-mode', type=int, choices=[2,3,4,5,6], default=3, help='number of vibronic modes for FC progression (allowed: 2,3,4,5,6)')
    parser.add_argument('--lineshape', type=str, default='gaussian')
    parser.add_argument('--beta', type=float, default=2.0)

    return parser

def main(args):

    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)

    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset, val_dataset, test_dataset, task_mean, task_std = load_dataset_splits(args)
    _log.info('Training set mean: {}, std:{}'.format(
        ' '.join(map(str, task_mean)), ' '.join(map(str, task_std))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_mean = torch.tensor(task_mean).to(device)
    task_std = torch.tensor(task_std).to(device)
    norm_factor = [task_mean, task_std]
    
    ''' Network '''
    args.out_channels = len(args.targets)
    args.task_mean = task_mean
    args.task_std = task_std
    model = equiformer_adapter.build(args)
    _log.info(model)
    model = model.to(device)
    
    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_exponential_decay(step, args),
    )
 
    criterion = None #torch.nn.L1Loss() # torch.nn.MSELoss() 
    if args.loss == 'MAE':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' Data Loader '''
    train_loader, val_loader, test_loader = build_dataloaders(args, train_dataset, val_dataset, test_dataset)

    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        train_err = train_one_step(
            model=model, criterion=criterion, norm_factor=norm_factor,
            data_loader=OneBatchLoader(batch),
            optimizer=optimizer, device=device,
            loss_scaler=(loss_scaler if 'loss_scaler' in locals() else None),
            clip_grad=getattr(args, 'clip_grad', None),
            input_step=step,
            print_freq=args.print_freq,
            loss_type=args.loss,
            spec_type=args.spectrum_type,
            line_shape=args.lineshape,
            beta=args.beta,
            logger=_log
        )
        lr_scheduler.step()

        if (step % args.eval_steps == 0) or (step == args.train_steps):
            val_mae, val_loss, _, _ = evaluate(
                model, norm_factor,
                val_loader, device,
                print_freq=args.print_freq, 
                loss_type=args.loss, 
                spec_type=args.spectrum_type,
                line_shape=args.lineshape,
                beta=args.beta,
                logger=_log
            )
            _log.info(f'[step {step}] val MAE={val_mae:.6f}, val loss={val_loss:.6f}, best val={best_val_loss:.6f}')

            test_mae, test_loss, preds, ids = evaluate(
                model, norm_factor,
                test_loader, device,
                print_freq=args.print_freq, 
                loss_type=args.loss, 
                spec_type=args.spectrum_type,
                line_shape=args.lineshape,
                beta=args.beta,
                logger=_log
            )
            _log.info(f'[step {step}] test MAE={test_mae:.6f}, test loss={test_loss:.6f}')

            if val_loss < best_val_loss and getattr(args, 'output_dir', None):
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                save_pred(preds.numpy(), ids, os.path.join(args.output_dir, 'pred.csv'), args.targets)
                save_spectrum(preds, ids, os.path.join(args.output_dir, 'p_spec.csv'), args.spectrum_type,
                        kernel_kind=args.lineshape, beta=args.beta)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(args.output_dir, 'checkpoint_best.ckpt'))

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()
    args.targets, args.standardize = build_spectrum_targets(args.spectrum_type, args.n_mode)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
