import torch

from services.models import Discriminator, Generator


def test_discriminator() -> None:
    """Функция проверяет размерность выходного тензора дискриминатора."""

    model = Discriminator()

    image: torch.Tensor = torch.rand((25, 3, 128, 128))
    result: torch.Tensor = torch.rand((25, 1))

    assert model(image).shape == result.shape


def test_generator() -> None:
    """Функция проверяет размерность выходного тензора генератора."""

    model = Generator()

    image: torch.Tensor = torch.rand((25, 3, 128, 128))

    assert model(image).shape == image.shape
