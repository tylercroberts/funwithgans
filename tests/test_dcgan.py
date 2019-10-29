import pytest
import io
import torch

@pytest.fixture
def sample_data():
    from dcgan.src.utils import get_data_loader
    return get_data_loader('tests/assets/', img_dim=64, batch_size=1, loader_workers=1)


@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def clargs():
    return {
            "storage-dir": "data",
            "model-dir": "models",
            "image-dir": "dcgan\\out",
            "log-dir": "logs",
            "reproducible": 0,
            "loader-workers": 2,
            "batch-size": 128,
            "image-dim": 64,
            "epochs": 1,
            "lr": 0.0002,
            "beta": 0.999,
            "ngpu": 1
    }

class TestDCGAN(object):

    def test_data_loader(self):
        from dcgan.src.utils import get_data_loader

        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=1, loader_workers=1)
        batch = next(iter(loader))
        assert list(batch[0].shape) == [1, 3, 64, 64]

        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=3, loader_workers=1)
        batch = next(iter(loader))
        assert list(batch[0].shape) == [3, 3, 64, 64]

    def test_plotting(self, device):
        from dcgan.src.utils import get_data_loader, plot_sample_images

        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=1, loader_workers=1)
        batch = next(iter(loader))

        img = plot_sample_images(device, batch, fig_size=(8, 8))
        buf = io.BytesIO()
        img.savefig(buf, format='png')
        buf.seek(0)
        assert "PNG" in str(buf.read())[:10], "File not successfully saved as image"



    def test_networks(self, device, sample_data):
        from dcgan.src.networks import Generator, Discriminator, LATENT_SHAPE
        from dcgan.src import weights_init

        try:
            gen = Generator(n_layers=3, image_dim=64, ngpu=1).to(device)
            gen.apply(weights_init)
        except:
            raise ValueError("Cannot apply weights_init to generator")

        try:
            disc = Discriminator(n_layers=3, image_dim=64, ngpu=1).to(device)
            disc.apply(weights_init)
        except:
            raise ValueError("Cannot apply weights_init to discriminator")

        real_batch = next(iter(sample_data))
        real_batch = real_batch[0].to(device)
        fake_batch = torch.randn(1, LATENT_SHAPE, 1, 1, device=device)

        try:
            fake = gen(fake_batch)
            assert fake is not None
        except AssertionError as e:
            raise ValueError("Generator unable to reconstruct image from noise.")

        try:
            decision = disc(real_batch)
            assert decision is not None
        except AssertionError as e:
            raise ValueError("Discriminator unable to make decision about image")

    def test_training_script(self, clargs):
        import argparse
        from dcgan.src.utils import parse_args

        arglist = [f"--{k}={v}" for k, v in clargs.items()]
        parsed_args = parse_args(arglist)

        assert isinstance(parsed_args, argparse.Namespace)

        with pytest.raises(ValueError):
            parse_args(['--config=randomdirthatprobablywontexistontestserver'])




