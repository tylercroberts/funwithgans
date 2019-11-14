
import logging
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pathlib import Path
from dcgan import DCGANModel


def train_model(loader: DataLoader, parameters: Dict[str, Any]) -> Dict[str, Any]:
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    logger = logging.getLogger(__name__)
    model = DCGANModel(parameters, channels=parameters['channels'], n_filters=parameters['n_filters'],
                       n_layers=parameters['n_layers'], train=True, seed=903,  # TODO: Fix reproducibility customization.
                       model_name='celeba')

    model.save_networks()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and parameters['ngpu'] > 0) else "cpu")
    noisy_vectors = torch.randn(64, parameters['latent_shape'], 1, 1, device=device)

    for epoch in range(parameters['epochs']):
        for i, data in enumerate(loader, 0):
            real = data[0].to(model.device)

            Dx, err_disc, err_gen, DGz1, DGz2 = model.update_parameters(real)

            if i % 50 == 0:
                logger.info(f"[{epoch}/{parameters['epochs']}][{i}/{len(loader)}]\t"
                            f"Loss - Discriminator: {round(err_disc.item(), 6)}\t"
                            f"Loss -  Generator: {round(err_gen.item(), 6)}\t"
                            f"D(x): {round(Dx, 6)}\t"
                            f"D(G(z)) Before update: {round(DGz1, 6)}\t"
                            f"D(G(z)) After update: {round(DGz2, 6)}")

                G_losses.append(err_gen.item())
                D_losses.append(err_disc.item())

            # Save training images from generator every 500 iters:
            if (iters % 500 == 0) or ((epoch == parameters['epochs']-1) and i == len(loader) - 1):

                img_list = _plot_training_images(model,
                                                 noisy_vectors, img_list, epoch, iters, parameters['image_dir'])
                iters += 1

        return {'img_list': img_list, 'lossG': G_losses, 'lossD': D_losses, 'model': model}


def _plot_training_images(model, noisy_vectors: torch.Tensor,
                          img_list: List,
                          epoch: int,
                          iters: int,
                          image_dir: Path
                          ) -> List:

    with torch.no_grad():
        fake = model.G(noisy_vectors).detach().cpu()

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    fig.suptitle(f"Images generated during training: Epoch{epoch} iter:{iters}")
    img = vutils.make_grid(fake, padding=2, normalize=True)
    img_list.append(img)
    plt.imshow(np.transpose(img[:64], (1, 2, 0)))
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.debug("Sample images successfully plotted")
    fig.savefig(Path(image_dir) / f"fake_{epoch}_{iters}.png")

    plt.close()

    return img_list


def plot_sample_images(loader: DataLoader, parameters: Dict[str, Any]) -> None:
    """
    Plot sample images from the dataset for sanity check.
    Does not save images, in order that users remember to close out the plot when finished with it.

    Args:
        batch (tuple): Element of `loader` from torch library
        parameters (dict): contains:
                device (Device or str): Device to move images to.
                fig_size (tuple): Size of image to plot.

    Returns:
        Figure object
    """
    device = torch.device("cuda:0" if (torch.cuda.is_available() and parameters['ngpu'] > 0) else "cpu")
    batch = next(iter(loader))
    fig = plt.figure(figsize=(np.sqrt(int(parameters['image_dim'])), np.sqrt(int(parameters['image_dim']))))
    plt.axis('off')
    fig.suptitle("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:parameters['image_dim']],
                                             padding=2, normalize=True).cpu(),
                            (1, 2, 0)))
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    fig.savefig(Path(parameters['image_dir']) / 'sample_real_images.png')
    log.info("Sample images successfully plotted")
