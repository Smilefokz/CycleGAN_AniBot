import torch
import logging

from services.models import Generator, Discriminator
from config.model_config import (
    TO_CUTE_ANIME_G_SELFIE_TO_ANIME, TO_CUTE_ANIME_G_ANIME_TO_SELFIE,
    TO_CUTE_ANIME_D_SELFIE, TO_CUTE_ANIME_D_ANIME,
    TO_SELFIE_G_SELFIE_TO_ANIME, TO_SELFIE_G_ANIME_TO_SELFIE,
    TO_SELFIE_D_SELFIE, TO_SELFIE_D_ANIME,
    TO_ANIME_G_SELFIE_TO_ANIME, TO_ANIME_G_ANIME_TO_SELFIE,
    TO_ANIME_D_SELFIE, TO_ANIME_D_ANIME
    )

# Инициализируем логгер:
logger = logging.getLogger(__name__)


# ------------------------------ Класс CycleGAN ------------------------------
# -------------------------------- Примечание --------------------------------
# Данный класс предназначен для загрузки весов предварительно обученной модели
# и не годится для обучения. Обучение модели производилось в Jupyter Notebook.
# Соответствующий файл для обучения модели располагается в директории по пути:
# services/train/cyclegan.ipynb
class CycleGAN:
    def __init__(self):

        self.G_selfie_to_anime = Generator()
        self.G_anime_to_selfie = Generator()

        self.D_selfie = Discriminator()
        self.D_anime = Discriminator()

    def load_weights(self, G_selfie_to_anime: str, G_anime_to_selfie: str,
                     D_selfie: str, D_anime: str) -> None:
        """
        Функция загружает необходимые веса для работы модели.

        Параметры:
        G_selfie_to_anime - путь до директории с весами
                            для генератора "из фотографии в аниме";
        G_anime_to_selfie - путь до директории с весами
                            для генератора "из аниме в фотографию";
        D_selfie - путь до директории с весами
                   для дискриминатора для "фотографий";
        D_anime - путь до директории с весами
                  для дискриминатора для "аниме".
        """
        self.G_selfie_to_anime.load_state_dict(
            torch.load(map_location=torch.device('cpu'), f=G_selfie_to_anime))
        self.G_anime_to_selfie.load_state_dict(
            torch.load(map_location=torch.device('cpu'), f=G_anime_to_selfie))
        self.D_selfie.load_state_dict(
            torch.load(map_location=torch.device('cpu'), f=D_selfie))
        self.D_anime.load_state_dict(
            torch.load(map_location=torch.device('cpu'), f=D_anime))


# Инициализация модели "из аниме в фотографию":
to_selfie_model = CycleGAN()
to_selfie_model.load_weights(TO_SELFIE_G_SELFIE_TO_ANIME,
                             TO_SELFIE_G_ANIME_TO_SELFIE,
                             TO_SELFIE_D_SELFIE,
                             TO_SELFIE_D_ANIME)

# Инициализация модели "из фотографии в аниме":
to_anime_model = CycleGAN()
to_anime_model.load_weights(TO_ANIME_G_SELFIE_TO_ANIME,
                            TO_ANIME_G_ANIME_TO_SELFIE,
                            TO_ANIME_D_SELFIE,
                            TO_ANIME_D_ANIME)

# Инициализация модели "из фотографии в милое аниме":
to_cute_anime_model = CycleGAN()
to_cute_anime_model.load_weights(TO_CUTE_ANIME_G_SELFIE_TO_ANIME,
                                 TO_CUTE_ANIME_G_ANIME_TO_SELFIE,
                                 TO_CUTE_ANIME_D_SELFIE,
                                 TO_CUTE_ANIME_D_ANIME)

# Выводим информацию об успешной инициализации CycleGAN:
logger.info('CycleGAN successfully initialized')
