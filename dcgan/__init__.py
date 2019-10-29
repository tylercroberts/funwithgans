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
    elif classname.find('ConvT') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANModel(BaseModel):
    def __init__(self, args, channels=3, n_filters=64, n_layers=3, train=True, logger=None, seed=903, model_name=''):
        super(DCGANModel, self).__init__(args, seed=seed, train=train, logger=logger)

        self.channels = channels
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.model_name = model_name
        self.criterion = nn.BCEWithLogitsLoss()
        self.real = None
        self.fake = None

        logger.info(f"DCGANModel Training Mode: {self.train}")

        gen = Generator(n_layers=self.n_layers, image_dim=self.image_dim, ngpu=self.ngpu).to(self.device)
        gen.apply(weights_init)
        self.models.append("G")
        self.G = gen

        if self.logger is not None:
            self.logger.info(f"{gen}")

        if train:
            disc = Discriminator(n_layers=self.n_layers, image_dim=self.image_dim, ngpu=self.ngpu).to(self.device)
            disc.apply(weights_init)
            self.models.append("D")
            self.D = disc

            if self.logger is not None:
                self.logger.info(f"{disc}")

            self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                                lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                lr=self.lr, betas=(self.beta1, 0.999))

    def forward(self, detach=False):
        fake = self.forward_G()
        self.fake = fake
        pos_preds, neg_preds = self.forward_D(self.real, self.fake.detach() if detach else self.fake)

        return fake, pos_preds, neg_preds

    def forward_G(self):
        """Run forward pass."""
        noise = torch.randn(self.batch_size, LATENT_SHAPE, 1, 1, device=self.device)
        fake = self.G(noise)
        return fake

    def forward_D(self, real, fake):
        return self.D(real), self.D(fake)

    def backward_G(self, neg_preds):

        label = torch.full((self.batch_size,),  # BATCH_SIZE array of 1s
                           fill_value=1, device=self.device)
        out = neg_preds.view(-1)
        err_gen = self.criterion(out, label)
        err_gen.backward()
        DGz2 = out.mean().item()
        self.optimizer_G.step()
        return err_gen, DGz2

    def backward_D(self, pos_preds, neg_preds):
        # --- Real Batch ---
        label = torch.full((self.batch_size,),  # BATCH_SIZE array of 1s
                           fill_value=1, device=self.device)
        err_disc_real = self.criterion(pos_preds, label)
        # Backward Pass
        err_disc_real.backward()  # Do each of the backwards separately to follow tip from optimizing GANs repo.
        Dx = pos_preds.mean().item()

        # --- Fake Batch ---
        label.fill_(0)
        err_disc_fake = self.criterion(neg_preds, label)
        # Backward pass:
        err_disc_fake.backward()
        DGz1 = neg_preds.mean().item()
        # Accumulate gradients
        err_disc = err_disc_real + err_disc_fake
        self.optimizer_D.step()

        return Dx, DGz1, err_disc

    def update_parameters(self, real=None):
        # Update Discriminators

        self.real = real
        fake, pos_preds, neg_preds = self.forward(detach=True)
        # self.set_requires_grad([self.D], True)
        self.D.zero_grad()
        Dx, DGz1, err_disc = self.backward_D(pos_preds.view(-1), neg_preds.view(-1))

        # Update Generators:
        # self.set_requires_grad([self.D], False)  # Don't need gradient on D to update G
        self.G.zero_grad()
        pos_preds, neg_preds = self.forward_D(self.real, self.fake)
        err_gen, DGz2 = self.backward_G(neg_preds)

        return Dx, err_disc, err_gen, DGz1, DGz2

    def load_networks(self):
        self.G = torch.load(self.model_dir / f"{self.model_name}_G.pt")

        if self.train:
            self.D = torch.load(self.model_dir / f"{self.model_name}_D.pt")

    def update_learning_rate(self):
        """Needed if using a scheduler"""
        pass

    def predict_test(self):
        with torch.no_grad():
            return self.forward()[0]


class GANLoss(nn.Module):

    def __init__(self, device, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # `register_buffer()` should be used to register variables that should *NOT* be trained by the optimizer.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.device = device
        self.loss = nn.BCEWithLogitsLoss()  # No need for sigmoid at the end of Discriminator Network with logits

    def _get_labels(self, preds, truth):
        out = self.real_label if truth else self.fake_label
        return out.expand_as(preds)

    def __call__(self, preds, truth):
        target = self._get_labels(preds, truth).to(self.device)
        return self.loss(preds, target)