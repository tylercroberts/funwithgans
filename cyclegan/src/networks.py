import torch
from torch import nn
from torchsummary import summary


#############
#
# Generator
#
#############


class Generator(nn.Module):
    """Difference between this generator and DCGAN is that there's a downsampling stage and then an upsampling stage"""
    def __init__(self, img_dim, channels, n_filters, use_dropout=False, stride=2, padding=1, bias=False, n_resnet=3,
                 n_downsampling=2, sample_kernel_size=3, resnet_kernel_size=3, outer_kernel_size=7, outer_padding=3):
        super(Generator, self).__init__()
        operations = list()

        # Initial Convolution block
        operations += [nn.ReflectionPad2d(outer_padding),
                       nn.Conv2d(channels, n_filters, kernel_size=outer_kernel_size, padding=0, bias=bias),
                       nn.BatchNorm2d(n_filters),
                       nn.LeakyReLU(0.2, inplace=True)],

        # Downsampling stages.
        for layer in range(n_downsampling):
            operations += [DownsampleCell(n_filters,
                                          layer,
                                          kernel_size=sample_kernel_size,
                                          stride=stride, padding=padding, bias=False)]

        # ResNet Stages:
        for layer in range(n_resnet):
            operations += [ResNetCell(n_filters, resnet_kernel_size, padding=1, bias=False, use_dropout=use_dropout)]

        # Upsampling Stages:
        # Because ResNetCells will preserve the shape, we will have the same number of upsampling steps
        for layer in range(n_downsampling):
            operations += [UpsampleCell(n_filters,
                                        layer,
                                        kernel_size=sample_kernel_size,
                                        stride=stride, padding=padding, bias=False,
                                        n_upsampling=n_downsampling)]

        # Final stages before output:
        operations += [nn.ReflectionPad2d(outer_padding),
                       nn.Conv2d(img_dim, channels, kernel_size=outer_kernel_size, padding=0),
                       nn.Tanh()]

        self.model = nn.Sequential(*operations)

    def forward(self, input):
        return self.model(input)

class ResNetCell(nn.Module):

    def __init__(self, img_dim, kernel_size=3, padding=1, bias=False, use_dropout=False):
        super(ResNetCell, self).__init__()
        self.cell = self.build_cell(img_dim, kernel_size, padding=padding, bias=bias, use_dropout=use_dropout)

    def build_cell(self, img_dim, kernel_size=3, padding=1, bias=False, use_dropout=False):
        operations = list()
        operations += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(img_dim, img_dim, kernel_size=kernel_size, padding=0, bias=bias),
                       nn.BatchNorm2d(img_dim),
                       nn.LeakyReLU(0.2, inplace=True)]

        if use_dropout:
            operations += [nn.Dropout(0.3)]

        operations += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(img_dim, img_dim, kernel_size=kernel_size, padding=0, bias=bias),
                       nn.BatchNorm2d(img_dim)]

        return nn.Sequential(*operations)

    def forward(self, input):
        return input + self.cell(input)

class DownsampleCell(nn.Module):
    def __init__(self, img_dim, layer_num, kernel_size, stride=2, padding=1, bias=False):
        super(DownsampleCell, self).__init__()
        self.cell = self.build_cell(img_dim, layer_num,
                                    kernel_size,
                                    stride=stride, padding=padding, bias=bias)

    def build_cell(self, img_dim, layer_num, kernel_size, stride=2, padding=1, bias=False):
        operations = list()

        # Need to divide max_mult by 2, because differently from DCGAN code, we're starting layer_num at 0
        ngf_mult = 2 ** layer_num

        operations += [nn.Conv2d(img_dim * ngf_mult,
                                 img_dim * ngf_mult * 2,
                                 kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias),
                       nn.BatchNorm2d(img_dim * ngf_mult * 2),
                       nn.LeakyReLU(0.2, inplace=True)]

        return nn.Sequential(*operations)

    def forward(self, input):
        return self.cell(input)


class UpsampleCell(nn.Module):
    def __init__(self, img_dim, layer_num, kernel_size, stride=2, padding=1, bias=False, n_upsampling=2):
        super(UpsampleCell, self).__init__()
        self.cell = self.build_cell(img_dim, layer_num,
                                    kernel_size,
                                    stride=stride, padding=padding, bias=bias, n_upsampling=n_upsampling)

    def build_cell(self, img_dim, layer_num, kernel_size, stride=2, padding=1, bias=False, n_upsampling=2):
        operations = list()
        # Need to divide max_mult by 2, because differently from DCGAN code, we're starting layer_num at 0
        ngf_mult = 2 ** (n_upsampling - layer_num)

        operations += [nn.ConvTranspose2d(img_dim * ngf_mult,
                                          img_dim * ngf_mult / 2,
                                          kernel_size=kernel_size,
                                          stride=stride, padding=padding, output_padding=padding, bias=bias),
                       nn.BatchNorm2d(img_dim * ngf_mult / 2),
                       nn.LeakyReLU(0.2, inplace=True)]

        return nn.Sequential(*operations)

    def forward(self, input):
        return self.cell(input)




##############
#
# Discriminator
#
##############


class Discriminator(nn.Module):
    """
    Input is 3 x 64 x 64 image
    Output is 1 x m * img_dim, where m is some positive integer. Can think of each element of representing
    the probability that a patch of the original image comes from the true distribution or not.

    Uses LeakyReLU instead of regular ReLU to help gradients flow through easier.

    """
    def __init__(self, n_layers, img_dim, ngpu, channels=3, stride=2, padding=1, bias=False, max_mult=8):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        operations = list()

        # Input cell here:
        operations += [nn.Conv2d(channels, img_dim, 4, stride=stride, padding=padding, bias=bias),
                       nn.LeakyReLU(0.2, inplace=True)]
        ndf_mult = 1

        # Main loop to add cells.
        for layer in range(1, n_layers):
            cell = ConvDiscriminatorCell(layer, img_dim=img_dim, kernel_size=4, stride=stride,
                                         padding=padding, bias=bias)
            operations += [cell]

        # Add final layer
        cell = ConvDiscriminatorCell(n_layers, img_dim=img_dim, kernel_size=4, stride=stride,
                                     padding=padding, bias=bias)
        operations += [cell]

        ndf_mult = min(2**n_layers, max_mult)

        # This is PatchGAN out (each of the n_layers * 8 elements in the output corresponds to a patch of image)
        operations += [nn.Conv2d(img_dim * ndf_mult, 1, 4, stride=1, padding=0, bias=bias)]

        self.main = nn.Sequential(*operations)

    def forward(self, input):
        return self.main(input)


class ConvDiscriminatorCell(nn.Module):

    def __init__(self, layer_num, img_dim=64, kernel_size=4, stride=2, padding=1,
                         bias=False):
        super(ConvDiscriminatorCell, self).__init__()
        self.cell = self.build_conv_cell(layer_num, img_dim=img_dim, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=bias)

    def build_conv_cell(self, layer_num, image_dim=64,  kernel_size=4, stride=2, padding=1,
                         bias=False, max_mult=8, img_dim=64):
        ndf_mult_prev = min(2**(layer_num-1), max_mult)
        ndf_mult = min(2**layer_num, max_mult)

        self.input_shape = ()

        return nn.Sequential(nn.Conv2d(img_dim * ndf_mult_prev,
                                       img_dim * ndf_mult,
                                       kernel_size,
                                       stride=stride, padding=padding, bias=bias),
                             nn.BatchNorm2d(img_dim * ndf_mult),
                             nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.cell(input)





