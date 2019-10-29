# Basic Imports
import itertools

# Torch Imports
import torch
from torch import nn

# Self Imports
from base.models import BaseModel
from dcgan.src.networks import Generator, Discriminator, LATENT_SHAPE


def weights_init(m):
    """ Basic weight initialization for DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANModel(BaseModel):
    def __init__(self, args, channels=3, n_filters=64, train=True, logger=None, model_name=''):
        super(DCGANModel, self).__init__(args, train=train, logger=logger)

        self.channels = channels
        self.n_filters = n_filters
        self.model_name = model_name
        self.criterion = GANLoss()

        gen = Generator(n_layers=channels, img_dim=self.img_dim, ngpu=self.ngpu).to(self.device)
        gen.apply(weights_init)
        self.models.append("G")
        self.G = gen

        if self.logger is not None:
            self.logger.info(f"{gen}")

        if train:
            disc = Discriminator(n_layers=3, img_dim=64, ngpu=self.ngpu).to(self.device)
            disc.apply(weights_init)
            self.models.append("D")
            self.D = disc

            if self.logger is not None:
                self.logger.info(f"{disc}")

            self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                                lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                lr=self.lr, betas=(self.beta1, 0.999))

    def forward(self, vector=None):
        """Run forward pass."""
        if vector is not None:
            fake = self.G(vector)
            return fake
        else:
            raise ValueError("Must pass value for vector")

    def backward_G(self, fake):
        self.G.zero_grad()
        out = self.D(fake).view(-1)  # Do another forward pass after update to D
        err_gen = self.criterion(out, False)
        err_gen.backward()
        DGz2 = out.mean().item()
        return DGz2

    def backward_D(self, data, fake):
        # --- Real Batch ---
        # Forward pass:
        real = data[0].to(self.device)
        b_size = real.size(0)
        out = self.D(real).view(-1)  # view(-1) is like np.ndarray.reshape(-1)
        err_disc_real = self.criterion(out, True)

        # Backward Pass
        err_disc_real.backward()
        Dx = out.mean().item()

        # --- Fake Batch ---
        # Forward pass:
        out = self.D(fake.detach()).view(-1)  # Need to understand why detach better
        err_disc_fake = self.criterion(out, False)

        # Backward pass:
        err_disc_fake.backward()
        DGz1 = out.mean().item()

        # Accumulate gradients
        err_disc = err_disc_real + err_disc_fake
        return Dx, DGz1, err_disc

    def update_params(self, real, noise):

        fake = self.forward(noise)

        # Update Discriminators
        self.set_requires_grad([self.D], True)
        self.optimizer_D.zero_grad()
        Dx, DGz1, err_disc = self.backward_D(real, fake)
        self.optimizer_D.step()

        # Update Generators:
        self.set_requires_grad([self.D], False)  # Don't need gradient on D to update G
        DGz2 = self.optimizer_G.zero_grad()

        self.backward_G(fake)
        self.optimizer_G.step()

        return Dx, err_disc, DGz1, DGz2

    def load_networks(self):
        self.G = torch.load(self.model_dir / f"{self.model_name}_G.pt")

        if self.train:
            self.D = torch.load(self.model_dir / f"{self.model_name}_D.pt")

    def update_learning_rate(self):
        """Needed if using a scheduler"""
        pass

    def predict_test(self, vector=None):
        if vector is not None:
            with torch.no_grad():
                self.forward(vector=vector)
        else:
            raise ValueError("No vector to turn into image passed")

class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        # `register_buffer()` should be used to register variables that should *NOT* be trained by the optimizer.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()  # No need for sigmoid at the end of Discriminator Network with logits

    def _get_labels(self, preds, truth):
        out = self.real_label if truth else self.fake_label
        return out.expand_as(preds)

    def __call__(self, preds, truth):
        target = self._get_labels(preds, truth)
        return self.loss(preds, target)