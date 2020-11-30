import os
import torch
from torch import nn
import torch.utils.data
from tqdm import tqdm

from torchvision.models.inception import inception_v3

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import linalg

import sys
from .inception import InceptionV3


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x


def get_activations(data_loader, model, device=None, batch_size=32, resize=False, n_samples=None):
  """Computes the activation of the given images

  Args:
      imgs: Torch dataset of (3xHxW) numpy images normalized in the
            range [-1, 1]
      cuda: whether or not to run on GPU
      batch_size: batch size for feeding into Inception v3
      splits: number of splits
  """
  try:
    n_batches = len(data_loader)
  except TypeError:  # data_loader can also be a generator object
    n_batches = float('inf')
  
  assert batch_size > 0
  if n_samples is not None:
    assert n_samples <= n_batches * batch_size
    n_batches = int(np.ceil(n_samples / batch_size))
  
  model = model.to(device)
  model.eval()
  up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
  
  def get_feat(x):
    with torch.no_grad():
      x = x.to(device)
      if resize:
        x = up(x)
      _, out = model(x)
      out = out[0].flatten(1, 3)
    return out.cpu().numpy()
  
  # Get predictions
  feat = []
  for batch in tqdm(data_loader, 'Compute activations', total=n_batches):
    if len(feat) >= n_batches:
      break
    if isinstance(batch, tuple) or isinstance(batch, list):  # img, label
      batch = batch[0]
    
    batch = batch.to(device)
    feat_i = get_feat(batch[:, :3])  # rgb only
    feat.append(feat_i)
  
  feat = np.concatenate(feat)
  if n_samples is not None:
    feat = feat[:n_samples]
    
  return feat


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                              ret_var=True, output=sys.stdout, **kernel_args):
  m = min(codes_g.shape[0], codes_r.shape[0])
  mmds = np.zeros(n_subsets)
  if ret_var:
    vars = np.zeros(n_subsets)
  choice = np.random.choice

  with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
    for i in bar:
      g = codes_g[choice(len(codes_g), min(m, subset_size), replace=False)]
      r = codes_r[choice(len(codes_r), min(m, subset_size), replace=False)]
      o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
      if ret_var:
        mmds[i], vars[i] = o
      else:
        mmds[i] = o
      bar.set_postfix({'mean': mmds[:i + 1].mean()})
  return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
  # use  k(x, y) = (gamma <x, y> + coef0)^degree
  # default gamma is 1 / dim
  X = codes_g
  Y = codes_r
  
  K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
  K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
  K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
  
  return _mmd2_and_variance(K_XX, K_XY, K_YY,
                            var_at_m=var_at_m, ret_var=ret_var)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
  # based on
  # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
  # but changed to not compute the full kernel matrix at once
  m = K_XX.shape[0]
  print(m, K_XX.shape, K_YY.shape, K_XY.shape)
  assert K_XX.shape == (m, m)
  assert K_XY.shape == (m, m)
  assert K_YY.shape == (m, m)
  if var_at_m is None:
    var_at_m = m
  
  # Get the various sums of kernels that we'll use
  # Kts drop the diagonal, but we don't need to compute them explicitly
  if unit_diagonal:
    diag_X = diag_Y = 1
    sum_diag_X = sum_diag_Y = m
    sum_diag2_X = sum_diag2_Y = m
  else:
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)
    
    sum_diag_X = diag_X.sum()
    sum_diag_Y = diag_Y.sum()
    
    sum_diag2_X = _sqn(diag_X)
    sum_diag2_Y = _sqn(diag_Y)
  
  Kt_XX_sums = K_XX.sum(axis=1) - diag_X
  Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
  K_XY_sums_0 = K_XY.sum(axis=0)
  K_XY_sums_1 = K_XY.sum(axis=1)
  
  Kt_XX_sum = Kt_XX_sums.sum()
  Kt_YY_sum = Kt_YY_sums.sum()
  K_XY_sum = K_XY_sums_0.sum()
  
  if mmd_est == 'biased':
    mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m))
  else:
    assert mmd_est in {'unbiased', 'u-statistic'}
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    if mmd_est == 'unbiased':
      mmd2 -= 2 * K_XY_sum / (m * m)
    else:
      mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))
  
  if not ret_var:
    return mmd2
  
  Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
  Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
  K_XY_2_sum = _sqn(K_XY)
  
  dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
  dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)
  
  m1 = m - 1
  m2 = m - 2
  zeta1_est = (
    1 / (m * m1 * m2) * (
    _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
    - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
    + 1 / (m * m * m1) * (
      _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
    - 2 / m ** 4 * K_XY_sum ** 2
    - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
    + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
  )
  zeta2_est = (
    1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
    - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
    + 2 / (m * m) * K_XY_2_sum
    - 2 / m ** 4 * K_XY_sum ** 2
    - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
    + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
  )
  var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
             + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)
  
  return mmd2, var_est


def _sqn(arr):
  flat = np.ravel(arr)
  return flat.dot(flat)


class KIDEvaluator(object):
  def __init__(self, device=None, batch_size=32, resize=False, n_samples=None, subset_size=1000):
    self.device = device
    self.batch_size = batch_size
    self.resize = resize
    self.n_samples = n_samples
    self.subset_size = subset_size
    
    self.init_model()
    
    self.act_target = None
  
  def init_model(self):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    self.model = InceptionV3([block_idx]).to(self.device)
    # model = inception_v3(pretrained=True, transform_input=False)
    # # replace fc layer by identity mapping to obtain features
    # model.fc = Identity()
    # return model
  
  def get_activations(self, data_loader):
    return get_activations(data_loader, self.model, device=self.device, batch_size=self.batch_size,
                          resize=self.resize, n_samples=self.n_samples)
  
  def initialize_target(self, target_loader, cache_file=None):
    if cache_file is not None:
      if os.path.isfile(cache_file):
        cache = np.load(cache_file)
        self.act_target = cache['act']
      else:
        self.act_target = self.get_activations(target_loader)
        np.savez(cache_file, act=self.act_target)
    else:
      self.act_target = self.get_activations(target_loader)
    
    if self.n_samples is None:
      self.n_samples = len(self.act_target)
  
  def is_initialized(self):
    return self.act_target is not None
  
  def get_kid(self, data_loader):
    assert self.is_initialized()
    act = self.get_activations(data_loader)
    return polynomial_mmd_averages(self.act_target, act, n_subsets=100, subset_size=self.subset_size)
