import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms


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


def _overwrite_args_with_config(args):
    with open(args.config, 'r') as f:
        json_config = json.loads(f.read())

    for k, v in json_config.items():
        args.__setattr__(k, v)

    return args


def parse_args(log):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config file which contains filepaths and other arguments',
                        default=False,
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

    parsed_args = parser.parse_args()
    if Path(parsed_args.config).exists():
        log.info(f"Overwriting any command-line arguments with config file found at "
                 f"{os.getcwd()}\\{parsed_args.config}")
        parsed_args = _overwrite_args_with_config(parsed_args)

    # Update the logger with filepath if necessary
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{parsed_args.log_dir}/celebagan_{time.time()}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    log.info("File logging successfully configured")

    for k, v in parsed_args.__dict__.items():
        log.debug(f"{k}: {v}")

    return parsed_args, log


def get_data_loader(data_path, img_dim, batch_size, loader_workers):
    transform_list = _get_transform_list(img_dim)
    images = dset.ImageFolder(root=str(data_path),
                              transform=transforms.Compose(transform_list))

    loader = torch.utils.data.DataLoader(images, batch_size=batch_size,
                                         shuffle=True, num_workers=loader_workers)

    return loader


def _get_transform_list(img_dim):
    transform_list = [transforms.Resize(img_dim),
                      transforms.CenterCrop(img_dim),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    return transform_list


def plot_sample_images(device, batch, fig_size=(8, 8)):
    fig = plt.figure(figsize=fig_size)
    plt.axis('off')
    fig.suptitle("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:fig_size[0]*fig_size[1]],
                                             padding=2, normalize=True).cpu(),
                            (1, 2, 0)))

    return fig
