import os
import numpy as np
import argparse

from model.resunet import ResUNetBN2C
from lib.data_loaders import collate_pair_fn
from train import get_trainer

import torch

from collections import defaultdict
import copy
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.datasets import mbes_data
from mbes_data.lib.benchmark_utils import to_tsfm

import wandb
import json

setup_seed(0)




def get_datasets(config: edict):
  if (config.dataset == 'multibeam'):
    train_set, val_set = mbes_data.get_multibeam_train_datasets(config)
    test_set = mbes_data.get_multibeam_test_datasets(config)
  else:
    raise NotImplementedError
  return train_set, val_set, test_set


def train(config):
  run = wandb.init(project='mbes', name='20230815-data-no-overlap-test',
                   config=config)
  wandb.tensorboard.patch(root_logdir='./20230815-data-no-overlap-test')
  train_set, val_set, test_set = get_datasets(config)
  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=collate_pair_fn,
    pin_memory=True,
    drop_last=True)
  val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=config.val_batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=collate_pair_fn,
    pin_memory=True,
    drop_last=True)
  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
    config=config,
    data_loader=train_loader,
    val_data_loader=val_loader
  )
  trainer.train()
  run.finish()




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mbes_config',
      type=str,
      default='mbes_data/configs/mbes_crop_train.yaml',
      help='Path to multibeam data config file')
  parser.add_argument(
      '--network_config',
      type=str,
      default='network_configs/train.yaml',
      help='Path to network config file')
  args = parser.parse_args()
  mbes_config = edict(load_config(args.mbes_config))
  network_config = edict(load_config(args.network_config))
  config = copy.deepcopy(mbes_config)
  for k, v in network_config.items():
    if k not in config:
      config[k] = v
  config['dataset_type'] = 'multibeam_npy_for_fcgf'
  os.makedirs(config.exp_dir, exist_ok=True)

  # Resume from checkpoint if available
  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/best_val_checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  print(config)
  train(config)