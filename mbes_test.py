import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features
from scripts.benchmark_util import run_ransac
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature_from_numpy

from model.resunet import ResUNetBN2C

import torch

from collections import defaultdict
import copy
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm
from mbes_data.datasets import mbes_data
from mbes_data.lib.evaluations import (compute_metrics, save_results_to_file, update_metrics_dict,
                                       summarize_metrics, print_metrics,
                                       ALL_METRICS_TEMPLATE, update_results)

setup_seed(0)

def draw_results(data, xyz_down_src, xyz_down_ref, feature_src, feature_ref, pred_trans):
  src_pcd = o3d.geometry.PointCloud()
  src_pcd.points = o3d.utility.Vector3dVector(xyz_down_src)

  src_pcd_trans = to_o3d_pcd(data['points_src'])
  src_pcd_trans.transform(pred_trans)
  print(f'pred transform: {pred_trans}')

  src_pcd_gt = to_o3d_pcd(data['points_src'])
  gt_trans = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
  src_pcd_gt.transform(gt_trans)
  print(f'gt trans: {gt_trans}')

  ref_pcd = o3d.geometry.PointCloud()
  ref_pcd.points = o3d.utility.Vector3dVector(xyz_down_ref)

  src_pcd = get_colored_point_cloud_feature(src_pcd,
                                            feature_src.detach().cpu().numpy(),
                                            config.voxel_size)
  ref_pcd = get_colored_point_cloud_feature(ref_pcd,
                                            feature_ref.detach().cpu().numpy(),
                                            config.voxel_size)
  src_pcd_trans.paint_uniform_color([1, 0, 0])
  src_pcd_gt.paint_uniform_color([0, 1, 0])
  o3d.visualization.draw_geometries([src_pcd, ref_pcd, src_pcd_trans, src_pcd_gt])


def get_datasets(config: edict):
  if (config.dataset == 'multibeam'):
    train_set, val_set = mbes_data.get_multibeam_train_datasets(config)
    test_set = mbes_data.get_multibeam_test_datasets(config)
  else:
    raise NotImplementedError

  return train_set, val_set, test_set


def test(config):
  # Setup logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler())
  logger.info('Start testing...')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load(config.model)
  model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  model = model.to(device)

  # Load data
  _, _, test_set = get_datasets(config)
  test_loader = DataLoader(
      test_set,
      batch_size=config.batch_size,
      num_workers=config.num_workers,
      shuffle=False)
  results = defaultdict(dict)
  for _, data in tqdm(enumerate(test_loader), total=len(test_set)):
    for key in data.keys():
      if isinstance(data[key], torch.Tensor):
        data[key] = data[key].squeeze(0)

    xyz_down_src, feature_src = extract_features(
        model=model,
        xyz=data['points_src'],
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)

    xyz_down_ref, feature_ref = extract_features(
        model=model,
        xyz=data['points_ref'],
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)

    pred_trans = run_ransac(
        to_o3d_pcd(xyz_down_src), to_o3d_pcd(xyz_down_ref),
        make_open3d_feature_from_numpy(feature_src.detach().cpu().numpy()),
        make_open3d_feature_from_numpy(feature_ref.detach().cpu().numpy()),
        config.voxel_size)
    update_results(results, data, pred_trans)

    if config.draw_registration_results:
      draw_results(data, xyz_down_src, xyz_down_ref,
                   feature_src, feature_ref, pred_trans)

  save_results_to_file(logger, results, config)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mbes_config',
      type=str,
      default='mbes_data/configs/mbesdata_test_meters.yaml',
      help='Path to multibeam data config file')
  parser.add_argument(
      '--network_config',
      type=str,
      default='config/network_configs/test.yaml',
      help='Path to network config file')
  args = parser.parse_args()
  mbes_config = edict(load_config(args.mbes_config))
  network_config = edict(load_config(args.network_config))
  config = copy.deepcopy(mbes_config)
  for k, v in network_config.items():
    if k not in config:
      config[k] = v
  os.makedirs(config.exp_dir, exist_ok=True)
  print(config)
  test(config)
