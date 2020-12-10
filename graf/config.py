import numpy as np
import torch
from torchvision.transforms import *

from .datasets import *
from .transforms import FlexGridRaySampler
from .utils import polar_to_cartesian, look_at, to_phi, to_theta


def save_config(outpath, config):
    from yaml import safe_dump
    with open(outpath, 'w') as f:
        safe_dump(config, f)


def update_config(config, unknown):
    # update config given args
    for idx,arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1,k2 = arg.replace("--","").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--','')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config


def get_data(config):
    H = W = imsize = config['data']['imsize']
    dset_type = config['data']['type']
    fov = config['data']['fov']

    transforms = Compose([
        Resize(imsize),
        ToTensor(),
        Lambda(lambda x: x * 2 - 1),
    ])

    kwargs = {
        'data_dirs': config['data']['datadir'],
        'transforms': transforms
    }

    if dset_type == 'carla':
        dset = Carla(**kwargs)

    elif dset_type == 'celebA':
        assert imsize <= 128, 'cropped GT data has lower resolution than imsize, consider using celebA_hq instead'
        transforms.transforms.insert(0, RandomHorizontalFlip())
        transforms.transforms.insert(0, CenterCrop(108))

        dset = CelebA(**kwargs)

    elif dset_type == 'celebA_hq':
        transforms.transforms.insert(0, RandomHorizontalFlip())
        transforms.transforms.insert(0, CenterCrop(650))

        dset = CelebAHQ(**kwargs)

    elif dset_type == 'cats':
      transforms.transforms.insert(0, RandomHorizontalFlip())
      dset = Cats(**kwargs)
  
    elif dset_type == 'cub':
        dset = CUB(**kwargs)

    dset.H = dset.W = imsize
    dset.focal = W/2 * 1 / np.tan((.5 * fov * np.pi/180.))
    radius = config['data']['radius']
    render_radius = radius
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(','))
        render_radius = max(radius)
    dset.radius = radius

    # compute render poses
    N = 40
    theta = 0.5 * (to_theta(config['data']['vmin']) + to_theta(config['data']['vmax']))
    angle_range = (to_phi(config['data']['umin']), to_phi(config['data']['umax']))
    render_poses = get_render_poses(render_radius, angle_range=angle_range, theta=theta, N=N)

    print('Loaded {}'.format(dset_type), imsize, len(dset), render_poses.shape, [H,W,dset.focal,dset.radius], config['data']['datadir'])
    return dset, [H,W,dset.focal,dset.radius], render_poses


def get_render_poses(radius, angle_range=(0, 360), theta=0, N=40, swap_angles=False):
    poses = []
    theta = max(0.1, theta)
    for angle in np.linspace(angle_range[0],angle_range[1],N+1)[:-1]:
        angle = max(0.1, angle)
        if swap_angles:
            loc = polar_to_cartesian(radius, theta, angle, deg=True)
        else:
            loc = polar_to_cartesian(radius, angle, theta, deg=True)
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        poses.append(RT)
    return torch.from_numpy(np.stack(poses))


def build_models(config, disc=True):
    from argparse import Namespace
    from submodules.nerf_pytorch.run_nerf_mod import create_nerf
    from .models.generator import Generator
    from .models.discriminator import Discriminator

    config_nerf = Namespace(**config['nerf'])
    # Update config for NERF
    config_nerf.chunk = min(config['training']['chunk'], 1024*config['training']['batch_size'])     # let batch size for training with patches limit the maximal memory
    config_nerf.netchunk = config['training']['netchunk']
    config_nerf.white_bkgd = config['data']['white_bkgd']
    config_nerf.feat_dim = config['z_dist']['dim']
    config_nerf.feat_dim_appearance = config['z_dist']['dim_appearance']

    render_kwargs_train, render_kwargs_test, params, named_parameters = create_nerf(config_nerf)

    bds_dict = {'near': config['data']['near'], 'far': config['data']['far']}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ray_sampler = FlexGridRaySampler(N_samples=config['ray_sampler']['N_samples'],
                                     min_scale=config['ray_sampler']['min_scale'],
                                     max_scale=config['ray_sampler']['max_scale'],
                                     scale_anneal=config['ray_sampler']['scale_anneal'],
                                     orthographic=config['data']['orthographic'])

    H, W, f, r = config['data']['hwfr']
    generator = Generator(H, W, f, r,
                          ray_sampler=ray_sampler,
                          render_kwargs_train=render_kwargs_train, render_kwargs_test=render_kwargs_test,
                          parameters=params, named_parameters=named_parameters,
                          chunk=config_nerf.chunk,
                          range_u=(float(config['data']['umin']), float(config['data']['umax'])),
                          range_v=(float(config['data']['vmin']), float(config['data']['vmax'])),
                          orthographic=config['data']['orthographic'],
                          )

    discriminator = None
    if disc:
        disc_kwargs = {'nc': 3,       # channels for patch discriminator
                       'ndf': config['discriminator']['ndf'],
                       'imsize': int(np.sqrt(config['ray_sampler']['N_samples'])),
                       'hflip': config['discriminator']['hflip']}

        discriminator = Discriminator(**disc_kwargs)

    return generator, discriminator


def build_lr_scheduler(optimizer, config, last_epoch=-1):
    import torch.optim as optim
    step_size = config['training']['lr_anneal_every']
    if isinstance(step_size, str):
        milestones = [int(m) for m in step_size.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch
        )
    return lr_scheduler
