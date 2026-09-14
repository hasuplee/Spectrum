"""Dataset/split loading and DataLoader construction shared by
train_PaiNN.py and train_Equiformer.py (Plan.md Step 5, G2).

Geoformer's Lightning-based DataModule (geoformer/data.py) is not touched
here (Plan.md non-goal: no forced merge with the Geoformer path).
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset.IrDB import IrDB, PtDB
import utils as utils
from common.training_utils import load_split_from_npz


def load_dataset_splits(args):
    """args needs: data_path, targets, split_index_npz, standardize, seed.
    Returns (train_dataset, val_dataset, test_dataset, task_mean, task_std)."""
    if args.data_path.startswith('IrDB'):
        dataset = IrDB(root=args.data_path, dataset_arg=args.targets)
    elif args.data_path.startswith('PtDB'):
        dataset = PtDB(root=args.data_path, dataset_arg=args.targets)
    else:
        raise Exception("data_path must be start 'IrDB' or 'PtDB'")

    idx_train, idx_val, idx_test = load_split_from_npz(args.split_index_npz)
    train_dataset = dataset[idx_train]
    val_dataset = dataset[idx_val]
    test_dataset = dataset[idx_test]

    # calculate dataset stats
    task_mean = [0.0 for _ in args.targets]
    task_std = [1.0 for _ in args.targets]
    if args.standardize:
        task_mean = [train_dataset.mean(i) for i in train_dataset.label_idx]
        task_std = [train_dataset.std(i) for i in train_dataset.label_idx]

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return train_dataset, val_dataset, test_dataset, task_mean, task_std


def build_dataloaders(args, train_dataset, val_dataset, test_dataset):
    """args needs: distributed, batch_size, workers, pin_mem.
    Returns (train_loader, val_loader, test_loader)."""
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem,
            drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem,
            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader
