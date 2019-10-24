import pytest
import io
import torch

@pytest.fixture
def device():
    return torch.device("cpu")


class TestDCGAN():

    def test_data_loader(self):
        from dcgan.src.utils import get_data_loader

        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=1, loader_workers=1)
        batch = next(iter(loader))
        assert list(batch[0].shape) == [1, 3, 64, 64]

        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=3, loader_workers=1)
        batch = next(iter(loader))
        assert list(batch[0].shape) == [3, 3, 64, 64]

    def test_plotting(self, device):
        from dcgan.src.utils import get_data_loader
        loader = get_data_loader('tests/assets/', img_dim=64, batch_size=1, loader_workers=1)
        batch = next(iter(loader))

        def _plot_sample_images(device):
            from dcgan.src.utils import plot_sample_images

            img = plot_sample_images(device, batch, fig_size=(8, 8))
            buf = io.BytesIO()
            img.savefig(buf, format='png')
            buf.seek(0)
            assert "PNG" in str(buf.read())[:10], "File not successfully saved as image"


        _plot_sample_images(device)

