import os
import time

import torch
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.optim import OptimWrapper

from .tools import get_parameter_number


def get_logger(record_save_dir, model, print_cfg, logger_name='mlcernet'):
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    files_save_dir = f'{record_save_dir}/{save_dir_date}'
    os.makedirs(f'{files_save_dir}/checkpoints', exist_ok=True)

    config_file = os.path.join(files_save_dir, 'config.py')
    print_cfg.dump(config_file)

    logger = MMLogger.get_instance(logger_name, log_file=f'{files_save_dir}/result.log')
    logger.info(f'total params: {get_parameter_number(model)}')
    logger.info('update params:')
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            logger.info(name)
    return logger, files_save_dir


def get_train_strategy(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cfg = dict(getattr(cfg, 'optim_wrapper', {}).get('optimizer', {}))
    optimizer_type = optimizer_cfg.pop('type', 'AdamW')
    optimizer_cfg.setdefault('lr', cfg.lr)
    optimizer_cfg.setdefault('weight_decay', cfg.weight_decay)

    optimizer_cls = getattr(torch.optim, optimizer_type)
    optimizer = optimizer_cls(params, **optimizer_cfg)
    optim_wrapper = OptimWrapper(optimizer=optimizer)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    return optim_wrapper, lr_scheduler


def build_evaluator(evaluator):
    if isinstance(evaluator, Evaluator):
        return evaluator
    return Evaluator(evaluator)
