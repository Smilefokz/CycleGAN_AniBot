import os
import logging

from pathlib import Path
from aiogram import Bot, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardRemove

from services.functions import style_image_func, bot_reaction_func
from lexicon.lexicon_ru import LEXICON_RU, LEXICON_BUTTONS_RU
from config.model_config import IN_PATH, OUT_PATH
from config.bot_config import Config, load_config
from keyboards.keyboards import (
    styles_keyboard,
    github_keyboard,
    help_keyboard,
    url_keyboard)

# Инициализируем логгер:
logger = logging.getLogger(__name__)

# Загружаем конфигуратор:
config: Config = load_config()

# Инициализируем бота и роутер:
bot = Bot(config.tg_bot.token)
router = Router()


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать на команду "/start", приветствовать
# пользователя и отправлять ему кнопку с инструкцией:
@router.message(CommandStart())
async def process_start_command(message: Message) -> None:
    await message.answer(text=LEXICON_RU['/start'],
                         reply_markup=help_keyboard)


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать на нажатие кнопки "help_button"
# и отправлять пользователю информацию из команды "/help":
@router.message(F.text == LEXICON_BUTTONS_RU['help_button'])
async def process_start_button(message: Message) -> None:
    await message.answer(text=LEXICON_RU['/help'],
                         reply_markup=ReplyKeyboardRemove())


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать на команду "/help"
# и знакомить пользователя с функционалом бота:
@router.message(Command(commands='help'))
async def process_help_command(message: Message) -> None:
    await message.answer(text=LEXICON_RU['/help'])


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать на команду "/download"
# и предлагать пользователю скачать фотографии для бота:
@router.message(Command(commands='download'))
async def process_download_command(message: Message) -> None:
    await message.answer(text=LEXICON_RU['/download'],
                         reply_markup=url_keyboard)


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать на команду "/github"
# и предлагать пользователю посетить GitHub проекта:
@router.message(Command(commands='github'))
async def process_github_command(message: Message) -> None:
    await message.answer(text=LEXICON_RU['/github'],
                         reply_markup=github_keyboard)


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать при получении фотографии
# и сохранять её в папку пользователя. Если пользователь
# впервые использует бота, будут созданы две личные папки -
# для приёма и отправки фотографий. Названия папок соответствуют
# id пользователя:
@router.message(F.photo)
async def save_image_func(message: Message) -> None:

    folder: int = message.from_user.id
    photo_name: int = message.message_id

    try:
        os.makedirs(f'{IN_PATH}/{folder}/')
        os.makedirs(f'{OUT_PATH}/{folder}/')
        # Выводим информацию о создании новых папок:
        logger.info(f"User {folder} has joined us!"
                    " New folders were added")
    except Exception:
        # Выводим сообщение, что папки уже созданы:
        logger.info("Folders with {folder} names"
                    " already exists")

    # Сохраняем присланное изображение в папку пользователя:
    await message.bot.download(
        file=message.photo[-1].file_id,
        destination=f'{IN_PATH}/{folder}/{photo_name}.jpg'
        )

    # Выводим информацию об успешном сохрании изображения:
    logger.info("Image successfully downloaded")

    # Присылаем пользователю клавиатуру styles_keyboard
    # и предлагаем ему выбрать желаемый стиль:
    await message.answer(
        text=LEXICON_RU['style'],
        reply_markup=styles_keyboard)


# -------------------------------------------------------------------------
# Данный хэндлер будет срабатывать при выборе пользователем
# желаемого стиля и оправлять ему преобразованное изображение:
@router.message(F.text.in_([LEXICON_BUTTONS_RU['to_cute_anime_button'],
                            LEXICON_BUTTONS_RU['to_selfie_button'],
                            LEXICON_BUTTONS_RU['to_anime_button']]))
async def transform_image_func(message: Message) -> None:

    folder: int = message.from_user.id

    media_group = style_image_func(folder, message)

    # Отправляем пользователю изображение:
    await bot.send_media_group(message.chat.id, media=media_group.build())

    # Удаляем клавиатуру styles_keyboard:
    await message.answer(
        text=bot_reaction_func(),
        reply_markup=ReplyKeyboardRemove())

    # Для экономии места на сервере удаляем все обработанные фотографии
    # в личных папках пользователя:
    for i in [f'{IN_PATH}/{folder}', f'{OUT_PATH}/{folder}']:
        [file.unlink() for file in Path(i).glob('*') if file.is_file()]

    # Выводим информацию об успешном удалении изображений:
    logger.info(f"All images from the {folder} folders"
                " have been deleted")
