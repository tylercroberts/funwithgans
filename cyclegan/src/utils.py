import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms


def logging_wrapper():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.info("Console logging successfully configured")

    return root


def set_flags(args, logger):
    LOADER_WORKERS = args.loader_workers
    BATCH_SIZE = args.batch_size
    IMG_DIM = args.image_dim
    EPOCHS = args.epochs
    LR = args.lr
    BETA1 = args.beta
    NGPU = args.ngpu
    logger.info(f"Script flags set are: loader_workers: {LOADER_WORKERS}, batch_size: {BATCH_SIZE},"
                f" img_dim: {IMG_DIM}, epochs: {EPOCHS}, lr: {LR}, beta1: {BETA1}, ngpu: {NGPU}")
    return LOADER_WORKERS, BATCH_SIZE, IMG_DIM, EPOCHS, LR, BETA1, NGPU



def _read_config_and_set_args(args):
    with open(args.config, 'r') as f:
        json_config = json.loads(f.read())

    arg_list = [f"--{k}={v}" for k, v in json_config.items()]
    args = parse_args(arg_list)

    return args


def parse_args(args, logger=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config file which contains filepaths and other arguments',
                        type=str)
    parser.add_argument("--storage-dir",
                        help="Storage directory for files, default: $CWD",
                        default=os.path.join(os.getcwd(), '..'))
    parser.add_argument("--data-dir",
                        help="Directory where necessary data is stored",
                        type=str)
    parser.add_argument("--model-dir",
                        help='Directory in which to store models after training',
                        default=os.path.join(os.getcwd(), '..', 'models'))
    parser.add_argument('--image-dir',
                        help='Directory in which to store images generated during training.',
                        default=os.path.join(os.getcwd(), 'dcgan', 'out'))
    parser.add_argument('--log-dir',
                        help='Directory in which to store log files.')
    parser.add_argument("--reproducible",
                        help='Flag to determine whether to set a predetermined seed for training',
                        type=bool)
    parser.add_argument("--loader-workers",
                        help='Workers to use for dataloading',
                        default=2,
                        type=int)
    parser.add_argument("--batch-size",
                        help='Batch size during training',
                        default=128,
                        type=int)
    parser.add_argument("--image-dim",
                        help='Size of images to generate',
                        default=64,
                        type=int)
    parser.add_argument("--epochs",
                        help='Number of times to iterate over dataset during training',
                        default=10,
                        type=int)
    parser.add_argument("--lr",
                        help='Learning rate for ADAM optimizer',
                        default=0.0002,
                        type=float)
    parser.add_argument("--beta",
                        help='Value to use for beta1 parameter for ADAM optimizer',
                        default=0.999,
                        type=float)
    parser.add_argument("--ngpu",
                        help='Number of times to iterate over dataset during training',
                        default=1,
                        type=int)

    parsed_args = parser.parse_args(args)
    try:
        if Path(parsed_args.config).exists():
            if logger is not None:
                logger.info(f"Using config file at: {parsed_args.config}")
            parsed_args = _read_config_and_set_args(parsed_args)

        else:
            raise ValueError("Config file specified was not found.")
    except TypeError:
        if logger is not None:
            logger.info("No config file specified. Using passed arguments instead")


def get_data_loader(data_path, img_dim, batch_size, loader_workers):
    transform_list = _get_transform_list(img_dim)
    real_images = dset.ImageFolder(root=str(data_path),
                                   transform=transforms.Compose(transform_list))

    real_loader = torch.utils.data.DataLoader(real_images, batch_size=batch_size,
                                              shuffle=True, num_workers=loader_workers)

    fake_images = dset.ImageFolder(root=str(data_path),
                                   transform=transforms.Compose(transform_list))

    fake_loader = torch.utils.data.DataLoader(fake_images, batch_size=batch_size,
                                              shuffle=True, num_workers=loader_workers)

    return real_loader, fake_loader


def _get_transform_list(img_dim):
    transform_list = [transforms.Resize(img_dim),
                      transforms.CenterCrop(img_dim),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    return transform_list


class ImagePool(object):
    """
    Used to train our generator on a bunch of previously generated fakes,
    since batch_size is going to be 1 mainly.
    """

    def __init__(self, size):
        self.size = size
        if self.size > 0:
            self.n_images = 0
            self.images = list()

    def get_images(self, images):
        if self.size == 0:
            return images

        out = list()
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            out = self.update_pool(image, out)

        out = torch.cat(out, 0)
        return out

    def update_pool(self, image, out):
        if self.n_images < self.size:
            self.n_images += 1
            self.images.append(image)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:
                random_id = random.randint(0, self.size - 1)
                tmp = self.images[random_id].clone()
                self.images[random_id] = image
                out.append(tmp)
            else:
                out.append(image)

        return out
