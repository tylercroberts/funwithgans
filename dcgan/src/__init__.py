from __future__ import print_function
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
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


# Self imports:
from dcgan.src.networks import Generator, Discriminator, LATENT_SHAPE
from dcgan.src.utils import logging_wrapper, parse_args, set_flags, get_data_loader, plot_sample_images


def weights_init(m):
    """ Basic weight initialization for DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    logger = logging_wrapper()

    args, logger = parse_args(logger)

    LOADER_WORKERS, BATCH_SIZE, IMG_DIM, EPOCHS, LR, BETA1, NGPU = set_flags(args, logger)

    random_seed = 903 if bool(args.reproducible) is True else random.randint(1, 100000)
    logger.info(f"Random seed is {random_seed}")
    random.seed(random_seed)

    # Root directory for dataset
    BASE_PATH = Path(args.storage_dir)
    DATA_PATH = BASE_PATH / 'data' / 'celeba' / 'img_align_celeba' if args.data_dir is None else Path(args.data_dir)
    MODEL_PATH = BASE_PATH / 'models' / 'dcgan' if args.model_dir is None else Path(args.model_dir)
    assert MODEL_PATH.exists()

    loader = get_data_loader(DATA_PATH, IMG_DIM, BATCH_SIZE, LOADER_WORKERS)
    logger.debug("Data loader successfully set up.")

    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
    logger.info(f"Device is set to {device}")

    real_batch = next(iter(loader))

    fig = plot_sample_images(device, real_batch)
    fig.savefig(Path(args.image_dir) / 'sample_real_images.png')
    logger.debug("Sample real images successfully plotted & saved.")
    plt.close(fig)

    logger.info("Successfully saved sample real images to disk.")
    gen = Generator(n_layers=3, img_dim=64, ngpu=NGPU).to(device)
    gen.apply(weights_init)
    logger.info(gen)

    disc = Discriminator(n_layers=3, img_dim=64, ngpu=NGPU).to(device)
    disc.apply(weights_init)
    logger.info(f"{disc}")

    criterion = nn.BCELoss()  # Binary cross-entropy

    # Noisy vectors below are used to see what sort of images the generator is creating throughout training.
    noisy_vectors = torch.randn(64, LATENT_SHAPE, 1, 1, device=device)

    pos_label = 1
    neg_label = 0

    optimizer_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    logger.info("Starting Training Loop...")

    for epoch in range(EPOCHS):
        # Iterate over batches
        for i, data in enumerate(loader, 0):

            # Update Discriminator to maximize log(D(x)) + log(1 - D(G(z)))

            # Using a tip from `ganhacks`, we'll do this in two steps
            # First, train an all-real batch, and store the gradient,
            # Then, train an all-fake batch, and accumulate the gradient with our previously stored one.
            # Use this accumulated gradient to update `D`

            disc.zero_grad()

            # --- Real Batch ---
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,),  # BATCH_SIZE array of 1s
                               pos_label, device=device)
            # Forward pass:
            out = disc(real_cpu).view(-1)  # Why view(-1)
            err_disc_real = criterion(out, label)
            err_disc_real.backward()
            Dx = out.mean().item()

            # --- Fake Batch ---
            noise = torch.randn(b_size, LATENT_SHAPE, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(neg_label)  # Replaces that arr of 1s with an arr of 0s
            out = disc(fake.detach()).view(-1)  # Need to understand why detach better
            err_disc_fake = criterion(out, label)
            err_disc_fake.backward()
            DGz1 = out.mean().item()

            # Accumulate gradients
            err_disc = err_disc_real + err_disc_fake

            optimizer_disc.step()

            # Update Generator to maximize log(D(G(z)))
            gen.zero_grad()
            label.fill_(pos_label)
            out = disc(fake).view(-1)  # Do another forward pass after update to D
            err_gen = criterion(out, label)
            err_gen.backward()
            DGz2 = out.mean().item()
            optimizer_gen.step()

            if i % 50 == 0:
                logger.info(f"[{epoch}/{EPOCHS}][{i}/{len(loader)}]\t"
                            f"Loss - Discriminator: {round(err_disc.item(), 6)}\t"
                            f"Loss -  Generator: {round(err_gen.item(), 6)}\t"
                            f"D(x): {round(Dx, 6)}\tD(G(z)): {round(DGz1, 6)}/{round(DGz2, 6)}")

            G_losses.append(err_gen.item())
            D_losses.append(err_disc.item())

            # Save training images from generator every 500 iters:
            if (iters % 500 == 0) or ((epoch == EPOCHS-1) and i == len(loader) - 1):
                with torch.no_grad():
                    fake = gen(noisy_vectors).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # When loading, remember to call `model.eval()` to set mode.
    torch.save(gen, MODEL_PATH / 'dcgan_gen.pt')
    logger.debug("Generator model successfully saved to disk")
    torch.save(disc, MODEL_PATH / 'dcgan_disc.pt')
    logger.debug("Discriminator model successfully saved to disk")

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(Path(args.image_dir) / 'sample_fake_images.gif')




