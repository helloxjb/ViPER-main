import numpy as np
import pandas as pd
import os
import random
import wandb

import torch
import argparse
import timm
import logging
import yaml

from stats import dataset_stats
from train_for_attention_diff import fit
from timm import create_model
from datasets import create_dataloader
from log import setup_default_logging
from models import VPT
from split import splits_AUROC, splits_F1

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):
    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'],
                           cfg['DATASET']['dataname'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir, 'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    dataname = cfg['DATASET']['dataname']
    splits = {
        'AUROC': splits_AUROC,
        'F1': splits_F1
    }
    for split_name, split in splits.items():
        for i in range(len(split[dataname])):
            # build Model
            if cfg['MODEL']['prompt_type']:
                model = VPT(
                    modelname=cfg['MODEL']['modelname'],
                    num_classes=cfg['DATASET']['num_classes'],
                    pretrained=True,
                    prompt_tokens=cfg['MODEL']['prompt_tokens'],
                    prompt_dropout=cfg['MODEL']['prompt_dropout'],
                    prompt_type=cfg['MODEL']['prompt_type']
                )
            else:
                model = create_model(
                    model_name=cfg['MODEL']['modelname'],
                    num_classes=cfg['DATASET']['num_classes'],
                    pretrained=True,
                )
            model.to(device)
            _logger.info('# of learnable params: {}'.format(
                np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

            # set training
            criterion = torch.nn.CrossEntropyLoss()
            # criterion = open_set_cross_entropy_loss()
            optimizer = __import__('torch.optim', fromlist='optim').__dict__[
                cfg['OPTIMIZER']['opt_name']](model.parameters(), **cfg['OPTIMIZER']['params'])

            # scheduler
            if cfg['TRAINING']['use_scheduler']:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg['TRAINING']['epochs'])
            else:
                scheduler = None
            item = i
            known = split[dataname][i]
            if dataname == 'cifar10' or dataname == 'svhn':
                unknown = list(set(list(range(0, 10)))-set(known))
            elif dataname == 'cifarplus10' or dataname == 'cifarplus50':
                known = split['cifar100'][i]
                unknown = split[dataname][i]
            elif dataname == 'tinyimagenet':
                unknown = list(set(list(range(0, 200)))-set(known))
            elif dataname == 'imgnr' or dataname == 'imgnc' or dataname == 'lsunr' or dataname == 'lsunc':
                unknown = [10]
            elif dataname == 'cifar10_svhn'or dataname=='cifar100_10':
                unknown = list(range(10))
            elif dataname == 'cifar10_100':
                unknown = list(range(100))
            elif dataname == 'imagenet30':
                unknown = list(range(10, 30))
            elif dataname == 'imagenet1k':
                unknown = list(range(100, 999))

            # load dataset
            trainset, testset, openset= __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
                datadir=cfg['DATASET']['datadir'],
                img_size=cfg['DATASET']['img_size'],
                mean=cfg['DATASET']['mean'],
                std=cfg['DATASET']['std'],
                num_samples_per_class=cfg['DATASET']['num_samples_per_class'],
                known=known,
                unknown=unknown
            )

            # load dataloader
            trainloader = create_dataloader(
                dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
            testloader = create_dataloader(
                dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)
            openloader = create_dataloader(
                dataset=openset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

            if cfg['TRAINING']['use_wandb']:
                # initialize wandb
                # os.environ["WANDB_MODE"] = "offline"
                os.environ["WANDB_API_KEY"] = ""
                # name
                wandb.init(name=f"{dataname}_{split_name}_split_{item}",
                           project='Visual Prompt Tuning', config=cfg)

            _logger.info(f"known: {known}")
            _logger.info(f"train images: {len(trainset)}")
            _logger.info(f"test images: {len(testset)}")
            _logger.info(f"open set images: {len(openset)}")

            # fitting model
            fit(cfg=cfg,
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                openloader=openloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=cfg['TRAINING']['epochs'],
                savedir=savedir,
                log_interval=cfg['TRAINING']['log_interval'],
                device=device,
                use_wandb=cfg['TRAINING']['use_wandb'])
            
            if cfg['TRAINING']['use_wandb']:
            # 完成当前wandb运行
                wandb.finish()
            break
        break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Prompt Tuning')
    parser.add_argument('--default_setting', type=str,
                        default=None, help='exp config file')
    parser.add_argument('--modelname', type=str, help='model name')
    parser.add_argument('--dataname', type=str, default='CIFAR10', choices=[
                        'CIFAR10', 'CIFAR100', 'SVHN', 'Tiny_ImageNet_200', 'cifar10', 'cifarplus10', 'cifarplus50', 'svhn', 'tinyimagenet', 'imagenet1k', 'imgnr', 'imgnc', 'lsunr', 'lsunc', 'cifar10_100','cifar100_10', 'cifar10_svhn','imagenet30'], help='data name')
    parser.add_argument('--img_resize', type=int,
                        default=None, help='Image Resize')
    parser.add_argument('--prompt_type', type=str,
                        choices=['shallow', 'deep'], help='prompt type')
    parser.add_argument('--prompt_tokens', type=int,
                        default=5, help='number of prompt tokens')
    parser.add_argument('--prompt_dropout', type=float,
                        default=0.0, help='prompt dropout rate')
    parser.add_argument('--no_wandb', action='store_false',
                        help='no use wandb')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.default_setting, 'r'), Loader=yaml.FullLoader)

    d_stats = dataset_stats[args.dataname.lower()]

    cfg['MODEL'] = {}
    cfg['MODEL']['modelname'] = args.modelname
    cfg['MODEL']['prompt_type'] = args.prompt_type
    cfg['MODEL']['prompt_tokens'] = args.prompt_tokens
    cfg['MODEL']['prompt_dropout'] = args.prompt_dropout
    cfg['DATASET']['num_classes'] = d_stats['num_classes']
    cfg['DATASET']['dataname'] = args.dataname
    cfg['DATASET']['img_size'] = args.img_resize if args.img_resize else d_stats['img_size']
    cfg['DATASET']['mean'] = d_stats['mean']
    cfg['DATASET']['std'] = d_stats['std']
    cfg['TRAINING']['use_wandb'] = args.no_wandb

    cfg['EXP_NAME'] = f"{args.modelname}-{args.prompt_type}-n_prompts{args.prompt_tokens}" if args.prompt_type else args.modelname

    run(cfg)
