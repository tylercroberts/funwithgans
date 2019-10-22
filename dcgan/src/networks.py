import torch.nn as nn

# Size of z latent vector (i.e. size of generator input)
LATENT_SHAPE = 100

# Number of channels in the training images. For color images this is 3
N_CHANNELS = 3

# Size of feature maps in generator
NFM_GENERATOR = 64

# Size of feature maps in discriminator
NFM_DISCRIMINATOR = 64


class Generator(nn.Module):
    """
    Input is a latent vector (think, right before output of discriminator)
    Output is a 3 x 64 x 64 image
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Takes in latent vector `z`
            # ConvTranspose2d takes in tensor of shape (N, C, H, W),
            # outputs shape (N, C_out, H_out, W_out) where:
            # H_out = (H_in - 1) * (stride_h - 2) * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1
            # W_out = (W_in - 1) * (stride_w - 2) * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1
            # Below, we'll reduce number of channels while increasing the H and W
            nn.ConvTranspose2d(LATENT_SHAPE,
                               NFM_GENERATOR * 8,
                               4,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(NFM_GENERATOR * 8),
            nn.ReLU(True),

            # After that cell, tensor is NFM_GENERATOR*8 x 4 x 4
            nn.ConvTranspose2d(NFM_GENERATOR * 8,
                               NFM_GENERATOR * 4,
                               4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_GENERATOR * 4),
            nn.ReLU(True),

            # After that cell, tensor is NFM_GENERATOR*4 x 8 x 8
            nn.ConvTranspose2d(NFM_GENERATOR * 4,
                               NFM_GENERATOR * 2,
                               4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_GENERATOR * 2),
            nn.ReLU(True),

            # After that cell, tensor is NFM_GENERATOR*2 x 16 x 16
            nn.ConvTranspose2d(NFM_GENERATOR * 2,
                               NFM_GENERATOR,
                               4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_GENERATOR),
            nn.ReLU(True),
            # After that cell, tensor is NFM_GENERATOR x 32 x 32
            nn.ConvTranspose2d(NFM_GENERATOR,
                               N_CHANNELS,
                               4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
            # Final shape is N_CHANNELS x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    Input is 3 x 64 x 64 image
    Output is binary decision, real or fake

    Uses LeakyReLU instead of regular ReLU to help gradients flow through easier.

    """
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is N_CHANNELS x 64 x 64, our base images.
            nn.Conv2d(N_CHANNELS,
                      NFM_DISCRIMINATOR,
                      4,
                      stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # After previous cell, tensor is NFM_DISCRIMINATOR x 32 x 32
            nn.Conv2d(NFM_DISCRIMINATOR,
                      NFM_DISCRIMINATOR * 2,
                      4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_DISCRIMINATOR * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # After previous cell, tensor is NFM_DISCRIMINATOR*2 x 16 x 16
            nn.Conv2d(NFM_DISCRIMINATOR * 2,
                      NFM_DISCRIMINATOR * 4,
                      4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_DISCRIMINATOR * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # After previous cell, tensor is NFM_DISCRIMINATOR * 4 x 8 x 8
            nn.Conv2d(NFM_DISCRIMINATOR * 4,
                      NFM_DISCRIMINATOR * 8,
                      4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NFM_DISCRIMINATOR * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # After previous cell, tensor is NFM_DISCRIMINATOR* 8 x 4 x 4
            nn.Conv2d(NFM_DISCRIMINATOR * 8,
                      1,
                      4,
                      stride=1, padding=0, bias=False),
            nn.Sigmoid()  # To get our final true/false pred.
        )

    def forward(self, input):
        return self.main(input)


