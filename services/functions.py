import os
import torch
import random

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from aiogram.types import Message, FSInputFile
from aiogram.utils.media_group import MediaGroupBuilder

from lexicon.lexicon_ru import LEXICON_BUTTONS_RU
from config.model_config import IN_PATH, OUT_PATH
from services.photo_processing import SelfieToAnimeDataset, transform
from services.cycle_gan import (
    to_cute_anime_model,
    to_selfie_model,
    to_anime_model)


# Функция-помощник для хэндлера с функцией transform_image_func:
def style_image_func(folder: int, message: Message) -> MediaGroupBuilder:
    """
    Данная функция преобразует изображение в выбранный пользователем стиль
    и сохраняет его в личной папке пользователя для отправки.

    Параметры:

    folder - личная папка пользователя;
    message - ответ пользователя.
    """

    # Создаём датасет с помощью класса SelfieToAnimeDataset:
    image_dataset = SelfieToAnimeDataset(f'{IN_PATH}/{folder}/')

    # Считаем количество полученных изображений:
    batch_size = len(os.listdir(f'{IN_PATH}/{folder}'))

    # Создаём объект класса DataLoader:
    image_loader = DataLoader(image_dataset, batch_size)
    iter_image = next(iter(image_loader))

    # Иницализируем билдер для полученных изображений:
    media_group = MediaGroupBuilder()

    # В зависимости от выбранного пользователем стиля
    # изменяем полученное изображение:
    with torch.no_grad():
        for i in range(batch_size):

            real = iter_image[i]

            if message.text == LEXICON_BUTTONS_RU['to_selfie_button']:
                fake = to_selfie_model.G_anime_to_selfie(real)
            elif message.text == LEXICON_BUTTONS_RU['to_anime_button']:
                fake = to_anime_model.G_selfie_to_anime(real)
            else:
                fake = to_cute_anime_model.G_selfie_to_anime(real)

            # Увеличививаем преобразованное изображение и сохраняем его:
            fake = transform(fake)
            save_image(fake * 0.5 + 0.5, (f'{OUT_PATH}/{folder}/{i}.jpg'))

            # Добавляем изображения в билдер:
            media_group.add_photo(
                media=FSInputFile(f'{OUT_PATH}/{folder}/{i}.jpg'))

    return media_group


def bot_reaction_func() -> str:
    """Функция присылает пользователю случайную реакцию бота."""

    reactions: list = [
        'Та-дааам!', 'А вот и я!', 'Попробуем снова!', 'Продолжим?',
        'Ещё разок?', 'Жду новых изображений :)', '^___^']

    return random.choice(reactions)
