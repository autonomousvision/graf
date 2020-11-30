import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')

# import ssl          # enable if downloading models gives CERTIFICATE_VERIFY_FAILED error
# ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples
from graf.transforms import ImgToPatch

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)

from external.colmap.filter_points import filter_ply


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--rotation_elevation', action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--shape_appearance', action='store_true', help='Create grid image showing shape/appearance variation.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')
    parser.add_argument('--reconstruction', action='store_true', help='Generate images and run COLMAP for 3D reconstruction.')

    args, unknown = parser.parse_known_args()
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    if args.pretrained:
        config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
        out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr, render_poses.shape)
    
    val_dataset = train_dataset                 # evaluate on training dataset for GANs
    if args.fid_kid:
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=False, sampler=None, drop_last=False   # enable shuffle for fid/kid computation
        )

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Get model file
    if args.pretrained:
        config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
        model_file = config_pretrained[config['data']['type']][config['data']['imsize']]
    else:
        model_file = 'model_best.pt'

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)
    
    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    # Evaluator
    evaluator = Evaluator(fid_kid, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()
    
    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    # Evaluation loop
    if args.fid_kid:
        # Specifically generate samples that can be saved
        n_samples = 1000
        ztest = zdist.sample((n_samples,))

        samples, _, _ = evaluator.create_samples(ztest.to(device))
        samples = (samples / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8)      # convert to unit8

        filename = 'samples_fid_kid_{:06d}.npy'.format(n_samples)
        outpath = os.path.join(eval_dir, filename)
        np.save(outpath, samples.numpy())
        print('Saved {} samples to {}.'.format(n_samples, outpath))

        samples = samples.to(torch.float) / 255

        n_vis = 8
        filename = 'fake_samples.png'
        outpath = os.path.join(eval_dir, filename)
        save_image(samples[:n_vis**2].clone(), outpath, nrow=n_vis)
        print('Plot examples under {}.'.format(outpath))

        filename = 'real_samples.png'
        outpath = os.path.join(eval_dir, filename)
        real = get_nsamples(val_loader, n_vis**2) / 2 + 0.5
        save_image(real[:n_vis ** 2].clone(), outpath, nrow=n_vis)
        print('Plot examples under {}.'.format(outpath))

        # Compute FID and KID
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)

        samples = samples * 2 - 1
        sample_loader = torch.utils.data.DataLoader(
            samples,
            batch_size=evaluator.batch_size, num_workers=config['training']['nworkers'],
            shuffle=False, pin_memory=False, sampler=None, drop_last=False
        )
        fid, kid = evaluator.compute_fid_kid(sample_loader)

        filename = 'fid_kid.csv'
        outpath = os.path.join(eval_dir, filename)
        with open(outpath, mode='w') as csv_file:
            fieldnames = ['fid', 'kid']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'fid': fid, 'kid': kid})

        print('Saved FID ({:.1f}) and KIDx100 ({:.2f}) to {}.'.format(fid, kid*100, outpath))

    if args.rotation_elevation:
        N_samples = 8
        N_poses = 20            # corresponds to number of frames
        render_radius = config['data']['radius']
        if isinstance(render_radius, str):  # use maximum radius
            render_radius = float(render_radius.split(',')[1])

        # compute render poses
        def get_render_poses_rotation_elevation(N_poses=float('inf')):
            """Compute equidistant render poses varying azimuth and polar angle, respectively."""
            range_theta = (to_theta(config['data']['vmin']), to_theta(config['data']['vmax']))
            range_phi = (to_phi(config['data']['umin']), to_phi(config['data']['umax']))

            theta_mean = 0.5 * sum(range_theta)
            phi_mean = 0.5 * sum(range_phi)

            N_theta = min(int(range_theta[1] - range_theta[0]), N_poses)  # at least 1 frame per degree
            N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

            render_poses_rotation = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=N_phi)
            render_poses_elevation = get_render_poses(render_radius, angle_range=range_theta, theta=phi_mean, N=N_theta,
                                                      swap_angles=True)

            return {'rotation': render_poses_rotation, 'elevation': render_poses_elevation}

        z = zdist.sample((N_samples,))

        for name, poses in get_render_poses_rotation_elevation(N_poses).items():
            outpath = os.path.join(eval_dir, '{}/'.format(name))
            os.makedirs(outpath, exist_ok=True)
            evaluator.make_video(outpath, z, poses, as_gif=False)
            torch.cuda.empty_cache()

    if args.shape_appearance:
        N_shapes = 5
        N_appearances = 5

        # constant pose
        pose = render_poses[len(render_poses) // 2]
        pose = pose.unsqueeze(0).expand(N_shapes * N_appearances, -1, -1)

        # sample shape latent codes
        z_shape = zdist.sample((N_shapes, 1))[..., :config['z_dist']['dim'] - config['z_dist']['dim_appearance']]
        z_shape = z_shape.expand(-1, N_appearances, -1)

        z_appearance = zdist.sample((1, N_appearances,))[..., config['z_dist']['dim_appearance']:]
        z_appearance = z_appearance.expand(N_shapes, -1, -1)

        z_grid = torch.cat([z_shape, z_appearance], dim=-1).flatten(0, 1)

        rgbs, _, _ = evaluator.create_samples(z_grid, poses=pose)
        rgbs = rgbs / 2 + 0.5

        outpath = os.path.join(eval_dir, 'shape_appearance.png')
        save_image(rgbs, outpath, nrow=N_shapes, padding=0)

    if args.reconstruction:

        N_samples = 8
        N_poses = 400            # corresponds to number of frames
        ztest = zdist.sample((N_samples,))

        # sample from mean radius
        radius_orig = generator_test.radius
        if isinstance(radius_orig, tuple):
            generator_test.radius = 0.5 * (radius_orig[0]+radius_orig[1])

        # output directories
        rec_dir = os.path.join(eval_dir, 'reconstruction')
        image_dir = os.path.join(rec_dir, 'images')
        colmap_dir = os.path.join(rec_dir, 'models')

        # generate samples and run reconstruction
        for i, z_i in enumerate(ztest):
            outpath = os.path.join(image_dir, 'object_{:04d}'.format(i))
            os.makedirs(outpath, exist_ok=True)

            # create samples
            z_i = z_i.reshape(1,-1).repeat(N_poses, 1)
            rgbs, _, _ = evaluator.create_samples(z_i.to(device))
            rgbs = rgbs / 2 + 0.5
            for j, rgb in enumerate(rgbs):
                save_image(rgb.clone(), os.path.join(outpath, '{:04d}.png'.format(j)))

            # run COLMAP for 3D reconstruction
            colmap_input_dir = os.path.join(image_dir, 'object_{:04d}'.format(i))
            colmap_output_dir = os.path.join(colmap_dir, 'object_{:04d}'.format(i))
            colmap_cmd = './external/colmap/run_colmap_automatic.sh {} {}'.format(colmap_input_dir, colmap_output_dir)
            print(colmap_cmd)
            os.system(colmap_cmd)

            # filter out white points
            filter_ply(colmap_output_dir)

        # reset radius for generator
        generator_test.radius = radius_orig
