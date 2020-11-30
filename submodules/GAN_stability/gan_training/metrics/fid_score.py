import os
import torch
from torch import nn
import torch.utils.data
from tqdm import tqdm

from torchvision.models.inception import inception_v3

import numpy as np
from scipy import linalg

import sys
from .inception import InceptionV3
from .kid_score import polynomial_mmd_averages


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, \
      'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
      'Training and test covariances have different dimensions'
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('FID has imaginary component {}. Set to "nan"'.format(m))
            return float('nan')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_activations(data_loader, model, device=None, batch_size=32, resize=False, n_samples=None):
    """Computes the inception score of the generated images imgs

    Args:
        imgs: Torch dataset of (3xHxW) numpy images normalized in the
              range [-1, 1]
        cuda: whether or not to run on GPU
        batch_size: batch size for feeding into Inception v3
        splits: number of splits
    """
    try:
        n_batches = len(data_loader)
    except TypeError:      # data_loader can also be a generator object
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
            out = out[0].flatten(1,3)
        return out.cpu().numpy()
    
    # Get predictions
    feat = []
    for batch in tqdm(data_loader, 'Compute statistics', total=n_batches):
        if len(feat) >= n_batches:
            break
        if isinstance(batch, tuple) or isinstance(batch, list):     # img, label
            batch = batch[0]

        batch = batch.to(device)
        feat_i = get_feat(batch[:, :3])      # rgb only
        feat.append(feat_i)
    
    feat = np.concatenate(feat)
    if n_samples is not None:
        feat = feat[:n_samples]

    return feat

def get_statistics(feat):
    
    # Now compute mean and std
    mu = np.mean(feat, axis=0)
    sigma = np.cov(feat, rowvar=False)
    
    return mu, sigma


def fid_score(data_loader1, data_loader2, device=None, batch_size=32, resize=False):
    mu1, sigma1 = get_statistics(data_loader1, device=device, batch_size=batch_size, resize=resize)
    mu2, sigma2 = get_statistics(data_loader2, device=device, batch_size=batch_size, resize=resize)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


class FIDEvaluator(object):
    def __init__(self, device=None, batch_size=32, resize=False, n_samples=None, n_samples_fake=1000, subset_size_kid=1000, subsets_kid=100):
        self.device = device
        self.batch_size = batch_size
        self.resize = resize
        self.n_samples = n_samples
        self.n_samples_fake = n_samples_fake
        self.subset_size_kid = subset_size_kid
        self.subsets_kid = subsets_kid
        
        self.init_model()

        self.mu_target = None
        self.sigma_target = None
        self.act_target = None
    
    def init_model(self):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(self.device)
        # model = inception_v3(pretrained=True, transform_input=False)
        # # replace fc layer by identity mapping to obtain features
        # model.fc = Identity()
        # return model
    
    def get_activations(self, data_loader, n_samples):
        return get_activations(data_loader, self.model, device=self.device, batch_size=self.batch_size,
                            resize=self.resize, n_samples=n_samples)

    def get_statistics(self, act):
        return get_statistics(act)
    
    def initialize_target(self, target_loader, cache_file=None, act_cache_file=None):
        if self.n_samples is None:
            self.n_samples = self.batch_size * len(target_loader)
        elif self.n_samples > self.batch_size * len(target_loader):
            print('WARNING: Total number of images smaller than %d, changing n_samples to %d!' % (self.n_samples, self.batch_size*len(target_loader)))
            self.n_samples = self.batch_size * len(target_loader)
    
        if act_cache_file is not None: # activation caches for KID
            if os.path.isfile(act_cache_file):
                cache = np.load(act_cache_file)
                self.act_target = cache['act']
            else:
                self.act_target = self.get_activations(target_loader, self.n_samples)
                np.savez(act_cache_file, act=self.act_target)

        if cache_file is not None:
            if os.path.isfile(cache_file):
                cache = np.load(cache_file)
                self.mu_target, self.sigma_target = cache['mu_target'], cache['sigma_target']
            else:
                self.mu_target, self.sigma_target = self.get_statistics(self.act_target)
                np.savez(cache_file, mu_target=self.mu_target, sigma_target=self.sigma_target)
        else:
            self.act_target = self.get_activations(target_loader, self.n_samples)
            self.mu_target, self.sigma_target = self.get_statistics(self.act_target)
          
    def is_initialized(self):
        return not any([self.mu_target is None, self.sigma_target is None, self.act_target is None])
    
    def get_fid(self, data_loader):
        assert self.is_initialized()
        act = self.get_activations(data_loader, self.n_samples_fake)
        mu, sigma = self.get_statistics(act)
        return calculate_frechet_distance(mu, sigma, self.mu_target, self.sigma_target)
  
    def get_kid(self, data_loader):
        assert self.is_initialized()
        act = self.get_activations(data_loader, self.n_samples_fake)
        return polynomial_mmd_averages(self.act_target, act, n_subsets=self.subsets_kid, subset_size=self.subset_size_kid)

    def get_fid_kid(self, data_loader):
        assert self.is_initialized()
        act = self.get_activations(data_loader, self.n_samples_fake)
        mu, sigma = self.get_statistics(act)
        fid = calculate_frechet_distance(mu, sigma, self.mu_target, self.sigma_target)
        kid = polynomial_mmd_averages(self.act_target, act, n_subsets=self.subsets_kid, subset_size=self.subset_size_kid)
        return fid, kid
