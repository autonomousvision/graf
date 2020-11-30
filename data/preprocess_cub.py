import argparse
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
from torchvision.transforms import CenterCrop


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Select split from CUB200-2011 and crop birds from images.'
    )
    parser.add_argument('root', type=str,
                        help='Path to data directory containing bounding box file and "images" folder.')
    parser.add_argument('maskdir', type=str, help='Path to data directory containing the segmentation masks.')
    args = parser.parse_args()

    imdir = os.path.join(args.root, 'images')
    bboxfile = os.path.join(args.root, 'bounding_boxes.txt')
    maskdir = args.maskdir
    namefile = './cub/filtered_files.txt'
    outdir = './cub'
    os.makedirs(outdir, exist_ok=True)

    # load files
    with open(namefile, 'r') as f:
        id_filename = [line.split(' ') for line in f.read().splitlines()]

    # load bounding boxes
    boxes = {}
    with open(bboxfile, 'r') as f:
        for line in f.read().splitlines():
            k, x, y, w, h = line.split(' ')
            box = float(x), float(y), float(x) + float(w), float(y) + float(h)  # (left, up, right, down)
            boxes[k] = box

    for i, (id, filename) in tqdm(enumerate(id_filename), total=len(id_filename)):
        path = os.path.join(imdir, filename)
        img = Image.open(path).convert('RGBA')

        # load alpha
        path = os.path.join(maskdir, filename.replace('.jpg', '.png'))
        alpha = Image.open(path)
        if alpha.mode == 'RGBA':
            alpha = alpha.split()[-1]
        alpha = alpha.convert('L')
        img.putalpha(alpha)

        # crop square images using bbox
        img = img.crop(boxes[id])
        s = max(img.size)
        img = CenterCrop(s)(img)             # CenterCrop pads image to square using zeros (also for alpha)

        # composite
        img = np.array(img)
        alpha = (img[..., 3:4]) > 127   # convert to binary mask
        bg = np.array(255 * (1. - alpha), np.uint8)
        img = img[..., :3] * alpha + bg
        img = Image.fromarray(img)

        img.save(os.path.join(outdir, '%06d.png' % i))

    print('Preprocessed {} images.'.format(len(glob.glob(os.path.join(outdir, '*.png')))))
