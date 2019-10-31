import torch.nn as nn

# Size of z latent vector (i.e. size of generator input)
LATENT_SHAPE = 100

# Number of channels in the training images. For color images this is 3
N_CHANNELS = 3

# Size of feature maps in generator
NFM_GENERATOR = 64

# Size of feature maps in discriminator
NFM_DISCRIMINATOR = 64


def build_gen_conv_cell(layer_num, ngf_mult_prev, kernel_size=4, stride=2, padding=1,
                        bias=False, max_mult=8, img_dim=64, n_layers=3):

    ngf_mult = min(2**(n_layers - layer_num), max_mult)

    return [nn.ConvTranspose2d(img_dim * ngf_mult_prev,
                               img_dim * ngf_mult,
                               kernel_size,
                               stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(img_dim * ngf_mult),
            nn.LeakyReLU(0.2, inplace=True)], ngf_mult


class Generator(nn.Module):
    """
    Input is a latent vector (think, right before output of discriminator)
    Output is a 3 x 64 x 64 image
    """
    def __init__(self, n_layers, image_dim, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        operations = list()
        # Takes in latent vector `z`
        # ConvTranspose2d takes in tensor of shape (N, C, H, W),
        # outputs shape (N, C_out, H_out, W_out) where:
        # H_out = (H_in - 1) * (stride_h - 2) * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1
        # W_out = (W_in - 1) * (stride_w - 2) * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1
        # Below, we'll reduce number of channels while increasing the H and W
        operations += [nn.ConvTranspose2d(LATENT_SHAPE,
                                          image_dim * 8,
                                          kernel_size=4,
                                          stride=1, padding=0, bias=False),
                       nn.BatchNorm2d(image_dim * 8),
                       nn.ReLU(inplace=True)]
        ngf_mult = 8
        for layer in range(1, n_layers):
            cell, ngf_mult = build_gen_conv_cell(layer, ngf_mult, kernel_size=4, stride=2, padding=1,
                                                 bias=False, max_mult=8, img_dim=NFM_GENERATOR, n_layers=n_layers)
            operations += cell

        # Add final layer
        cell, nfg_mult = build_gen_conv_cell(n_layers, ngf_mult, kernel_size=4, stride=2, padding=1,
                                             bias=False, max_mult=8, img_dim=NFM_GENERATOR, n_layers=n_layers)
        operations += cell

        operations += [nn.ConvTranspose2d(image_dim,
                                          N_CHANNELS,
                                          4,
                                          stride=2, padding=1, bias=False),
                       nn.Tanh()]

        self.main = nn.Sequential(*operations)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    Input is 3 x 64 x 64 image
    Output is 1 x m * img_dim, where m is some positive integer. Can think of each element of representing
    the probability that a patch of the original image comes from the true distribution or not.

    Uses LeakyReLU instead of regular ReLU to help gradients flow through easier.

    """
    def __init__(self, n_layers, image_dim, ngpu, stride=2, padding=1, bias=False, max_mult=8):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        operations = list()

        # Input cell here:
        operations += [nn.Conv2d(N_CHANNELS, image_dim, 4, stride=stride, padding=padding, bias=bias),
                       nn.LeakyReLU(0.2, inplace=True)]
        ndf_mult = 1

        # Main loop to add cells.
        for layer in range(1, n_layers):
            cell = ConvDiscriminatorCell(layer, image_dim=image_dim, kernel_size=4, stride=stride,
                                         padding=padding, bias=bias)
            operations += [cell]

        # Add final layer
        cell = ConvDiscriminatorCell(n_layers, image_dim=image_dim, kernel_size=4, stride=stride,
                                     padding=padding, bias=bias)
        operations += [cell]

        ndf_mult = min(2**n_layers, max_mult)

        # This is PatchGAN out (each of the n_layers * 8 elements in the output corresponds to a patch of image)
        operations += [nn.Conv2d(image_dim * ndf_mult, 1, 4, stride=1, padding=0, bias=bias)]

        self.main = nn.Sequential(*operations)

    def forward(self, input):
        return self.main(input)


class ConvDiscriminatorCell(nn.Module):

    def __init__(self, layer_num, image_dim=64, kernel_size=4, stride=2, padding=1,
                 bias=False):
        super(ConvDiscriminatorCell, self).__init__()
        self.cell = self.build_conv_cell(layer_num, image_dim=image_dim, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=bias)

    def build_conv_cell(self, layer_num, image_dim=64,  kernel_size=4, stride=2, padding=1,
                        bias=False, max_mult=8, img_dim=64):
        ndf_mult_prev = min(2**(layer_num-1), max_mult)
        ndf_mult = min(2**layer_num, max_mult)

        return nn.Sequential(nn.Conv2d(image_dim * ndf_mult_prev,
                                       image_dim * ndf_mult,
                                       kernel_size,
                                       stride=stride, padding=padding, bias=bias),
                             nn.BatchNorm2d(image_dim * ndf_mult),
                             nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.cell(input)
