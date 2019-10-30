import sys
import time
import torch
import random
import logging
import numpy as np
import matplotlib.animation as animation

from pathlib import Path
import matplotlib.pyplot as plt

import torchvision.utils as vutils


from dcgan import DCGANModel
from dcgan.src.utils import parse_args, logging_wrapper, get_data_loader, plot_sample_images, set_flags
from dcgan.src.networks import LATENT_SHAPE

if __name__ == '__main__':
    logger = logging_wrapper()
    args = parse_args(sys.argv[1:])

    # Update the logger with filepath if necessary
    if logger is not None:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(f"{args.log_dir}/celebagan_{time.time()}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.info("File logging successfully configured")

    if logger is not None:
        for k, v in args.__dict__.items():
            logger.debug(f"{k}: {v}")

    LOADER_WORKERS, BATCH_SIZE, IMG_DIM, EPOCHS, LR, BETA1, NGPU = set_flags(args, logger)

    random_seed = 903 if bool(args.reproducible) is True else random.randint(1, 100000)

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

    model = DCGANModel(args, channels=3, n_filters=64, n_layers=3, train=True, logger=logger, seed=random_seed,
                       model_name='celeba')

    logger.info("DCGANModel successfully initialized")

    # Noisy vectors below are used to see what sort of images the generator is creating throughout training.
    noisy_vectors = torch.randn(64, LATENT_SHAPE, 1, 1, device=device)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    logger.info("Starting Training Loop...")

    for epoch in range(EPOCHS):
        for i, data in enumerate(loader, 0):
            real = data[0].to(model.device)

            Dx, err_disc, err_gen, DGz1, DGz2 = model.update_parameters(real)

            if i % 50 == 0:
                logger.info(f"[{epoch}/{EPOCHS}][{i}/{len(loader)}]\t"
                            f"Loss - Discriminator: {round(err_disc.item(), 6)}\t"
                            f"Loss -  Generator: {round(err_gen.item(), 6)}\t"
                            f"D(x): {round(Dx, 6)}\t"
                            f"D(G(z)) Before update: {round(DGz1, 6)}\t"
                            f"D(G(z)) After update: {round(DGz2, 6)}")

                G_losses.append(err_gen.item())
                D_losses.append(err_disc.item())

            # Save training images from generator every 500 iters:
            if (iters % 500 == 0) or ((epoch == EPOCHS-1) and i == len(loader) - 1):
                with torch.no_grad():
                    fake = model.G(noisy_vectors).detach().cpu()

                fig = plt.figure(figsize=(8, 8))
                plt.axis('off')
                fig.suptitle(f"Images generated during training: Epoch{epoch} iter:{iters}")
                img = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(img)
                plt.imshow(np.transpose(img[:64], (1, 2, 0)))
                fig.savefig(Path(args.image_dir) / f"fake_{epoch}_{iters}.png")
                plt.close()
            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(args.imgae_dir) / 'training_loss.png')
    plt.show()


    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(Path(args.image_dir) / 'sample_fake_images.gif')
