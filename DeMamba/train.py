import random
import argparse
import yaml
import torch
import numpy as np
import os
import util
from util import build_model, train_one_epoch
from dataloader import generate_dataset_loader_from_json
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR


def parse_device_ids(value):
    """Parse explicit CUDA device IDs, e.g. ``0`` or ``0,1,...,7``."""
    try:
        device_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('--device-ids must be comma-separated integers') from error
    if not device_ids or len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise argparse.ArgumentTypeError('--device-ids must contain unique non-negative integers')
    return device_ids


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of detector in yaml format')
    parser.add_argument('--device-ids', default='0',
                        help='CUDA device IDs: 0 for one GPU, or 0,1,2,3,4,5,6,7 for eight GPUs')
    parser.add_argument('--train-batch-size', type=int, default=None,
                        help='Optional override for cfg.train_batch_size')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Optional override for cfg.val_batch_size')
    parser.add_argument('--max-epoch', type=int, default=None,
                        help='Optional override for cfg.max_epoch')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed; use the same seed for baseline and neuron method')
    args = parser.parse_args()
    args.device_ids = parse_device_ids(args.device_ids)
    return args


if __name__ == '__main__': 
    args = get_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    for key, value in (('train_batch_size', args.train_batch_size),
                       ('val_batch_size', args.val_batch_size),
                       ('max_epoch', args.max_epoch)):
        if value is not None:
            if value <= 0:
                raise ValueError(f'--{key.replace("_", "-")} must be positive')
            cfg[key] = value
    print(f"******* Random seed: {args.seed} *******")
    print("******* Building models. *******")
    print(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current DeMamba training loop')
    available = torch.cuda.device_count()
    if max(args.device_ids) >= available:
        raise ValueError(f'Requested --device-ids {args.device_ids}, but only {available} CUDA device(s) are visible')
    primary_device = args.device_ids[0]
    torch.cuda.set_device(primary_device)
    print(f"******* Using CUDA device IDs: {args.device_ids} (primary cuda:{primary_device}) *******")
    model = util.build_model(
        cfg['model'],
        neuron_indices_path=cfg.get('neuron_indices_path'),
        xclip_model_path=cfg.get('xclip_model_path'),
    )
    model = model.to(torch.device(f'cuda:{primary_device}'))

    if cfg['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    if len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=primary_device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying tuning_mode")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[7], gamma=0.3)
    if cfg['mode'] == 'binary':
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()
    
    trMaxEpoch = cfg['max_epoch']
    snapshot_path = cfg['save_dir']
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    max_epoch, max_acc = 0, 0

    for epochID in range(0, trMaxEpoch):
        print("******* Training epoch", str(epochID)," *******")
        print("******* Building datasets. *******")
        train_loader, val_loader, test_fake_segments = generate_dataset_loader_from_json(cfg)
        max_epoch, max_acc, epoch_time = train_one_epoch(cfg, model, loss, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path, test_fake_segments)
        print("******* Ending epoch", str(epochID)," Time ", str(epoch_time), "*******")
