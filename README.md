# GRAF

<div style="text-align: center">
<img src="animations/carla_256.gif" width="512"/><br>
</div>

This repository contains official code for the paper
[GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips).

You can find detailed usage instructions for training your own models and using pre-trained models below.


If you find our code or paper useful, please consider citing

    @inproceedings{Schwarz2020NEURIPS,
      title = {GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis},
      author = {Schwarz, Katja and Liao, Yiyi and Niemeyer, Michael and Geiger, Andreas},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2020}
    }

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `graf` using
```
conda env create -f environment.yml
conda activate graf
```

Next, for nerf-pytorch install torchsearchsorted. Note that this requires `torch>=1.4.0` and `CUDA >= v10.1`.
You can install torchsearchsorted via
``` 
cd submodules/nerf_pytorch
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ../../../
```

## Demo

You can now test our code via:
```
python eval.py configs/carla.yaml --pretrained --rotation_elevation
```
This script should create a folder `results/carla_128_from_pretrained/eval/` where you can find generated videos varying camera pose for the Cars dataset.

## Datasets

If you only want to generate images using our pretrained models you do not need to download the datasets.
The datasets are only needed if you want to train a model from scratch.

### Cars

To download the Cars dataset from the paper simply run
```
cd data
./download_carla.sh
cd ..
```
This creates a folder `data/carla/` downloads the images as a zip file and extracts them to `data/carla/`. 
While we do <em>not</em> use camera poses in this project we provide them for completeness. Your can download them by running
```
cd data
./download_carla_poses.sh
cd ..
```
This downloads the camera intrinsics (single file, equal for all images) and extrinsics corresponding to each image.  

### Faces

Download [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Then replace `data/celebA` in `configs/celebA.yaml` with `*PATH/TO/CELEBA*/Img/img_align_celebA`.

Download [celebA_hq](https://github.com/tkarras/progressive_growing_of_gans).
Then replace `data/celebA_hq` in `configs/celebAHQ.yaml` with `*PATH/TO/CELEBA_HQ*`.

### Cats
Download the [CatDataset](https://www.kaggle.com/crawford/cat-dataset).
Run
```
cd data
python preprocess_cats.py PATH/TO/CATS/DATASET
cd ..
```
to preprocess the data and save it to `data/cats`.
If successful this script should print: `Preprocessed 9407 images.`

### Birds
Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and the corresponding [Segmentation Masks](https://drive.google.com/file/d/1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP/view).
Run
```
cd data
python preprocess_cub.py PATH/TO/CUB-200-2011 PATH/TO/SEGMENTATION/MASKS
cd ..
```
to preprocess the data and save it to `data/cub`.
If successful this script should print: `Preprocessed 8444 images.`

## Usage

When you have installed all dependencies, you are ready to run our pre-trained models for 3D-aware image synthesis.

### Generate images using a pretrained model

To evaluate a pretrained model, run 
```
python eval.py CONFIG.yaml --pretrained --fid_kid --rotation_elevation --shape_appearance
```
where you replace CONFIG.yaml with one of the config files in `./configs`. 

This script should create a folder `results/EXPNAME/eval` with FID and KID scores in `fid_kid.csv`, videos for rotation and elevation in the respective folders and an interpolation for shape and appearance, `shape_appearance.png`. 

Note that some pretrained models are available for different image sizes which you can choose by setting `data:imsize` in the config file to one of the following values:
```
configs/carla.yaml: 
    data:imsize 64 or 128 or 256 or 512
configs/celebA.yaml:
    data:imsize 64 or 128
configs/celebAHQ.yaml:
    data:imsize 256 or 512
```

### Train a model from scratch

To train a 3D-aware generative model from scratch run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with your config file.
The easiest way is to use one of the existing config files in the `./configs` directory 
which correspond to the experiments presented in the paper. 
Note that this will train the model from scratch and will not resume training for a pretrained model.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./monitoring --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.

### Evaluation of a new model

For evaluation of the models run
```
python eval.py CONFIG.yaml --fid_kid --rotation_elevation --shape_appearance
```
where you replace `CONFIG.yaml` with your config file.

## Multi-View Consistency Check

You can evaluate the multi-view consistency of the generated images by running a Multi-View-Stereo (MVS) algorithm on the generated images. This evaluation uses [COLMAP](https://colmap.github.io/) and make sure that you have COLMAP installed to run
```
python eval.py CONFIG.yaml --reconstruction
```
where you replace `CONFIG.yaml` with your config file. You can also evaluate our pretrained models via:
```
python eval.py configs/carla.yaml --pretrained --reconstruction
```
This script should create a folder `results/EXPNAME/eval/reconstruction/` where you can find generated multi-view images in `images/` and the corresponding 3D reconstructions in `models/`.

## Further Information

### GAN training

This repository uses Lars Mescheder's awesome framework for [GAN training](https://github.com/LMescheder/GAN_stability).

### NeRF

We base our code for the Generator on this great [Pytorch reimplementation](https://github.com/yenchenlin/nerf-pytorch) of Neural Radiance Fields.
