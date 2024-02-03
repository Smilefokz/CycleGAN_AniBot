from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.types import (
    KeyboardButton, ReplyKeyboardMarkup,
    InlineKeyboardButton, InlineKeyboardMarkup)

from lexicon.lexicon_ru import LEXICON_BUTTONS_RU
from config.model_config import (
    SELFIE_URL, ANIME_URL,
    CUTE_ANIME_URL, GIT_URL)


# ---------------------- Клавиатура для кнопки-помощника ----------------------
# Создаём кнопку для вызова инструкции:
help_button = KeyboardButton(
    text=LEXICON_BUTTONS_RU['help_button'])

# Иницализируем билдер для данной клавиатуры:
help_kb_builder = ReplyKeyboardBuilder()

# Добавляем кнопку в билдер:
help_kb_builder.row(help_button, width=1)

# Создаём help_keyboard клавиатуру:
help_keyboard: ReplyKeyboardMarkup = help_kb_builder.as_markup(
    resize_keyboard=True)


# ------------------- Клавиатура для скачивания изображений -------------------
# Создаём кнопку для скачивания фотографий:
url_button_selfie = InlineKeyboardButton(
    text=LEXICON_BUTTONS_RU['url_button_selfie'],
    url=SELFIE_URL)

# Создаём кнопку для скачивания аниме изображений:
url_button_anime = InlineKeyboardButton(
    text=LEXICON_BUTTONS_RU['url_button_anime'],
    url=ANIME_URL)

# Создаём кнопку для скачивания аниме изображений:
url_button_cute_anime = InlineKeyboardButton(
    text=LEXICON_BUTTONS_RU['url_button_cute_anime'],
    url=CUTE_ANIME_URL)

# Иницализируем билдер для данной клавиатуры:
url_kb_builder = InlineKeyboardBuilder()

# Добавляем кнопки в билдер:
url_kb_builder.row(
    url_button_selfie,
    url_button_anime,
    url_button_cute_anime,
    width=1)

# Создаём url_keyboard клавиатуру:
url_keyboard: InlineKeyboardMarkup = url_kb_builder.as_markup(
    resize_keyboard=True)


# ----------- Клавиатура для перехода на страницу проекта в GitHub -----------
# Создаём кнопку для перехода в GitHub:
url_button_github = InlineKeyboardButton(
    text=LEXICON_BUTTONS_RU['url_button_github'],
    url=GIT_URL)

# Иницализируем билдер для данной клавиатуры:
github_kb_builder = InlineKeyboardBuilder()

# Добавляем кнопки в билдер:
github_kb_builder.row(url_button_github, width=1)

# Создаём url_keyboard клавиатуру:
github_keyboard: InlineKeyboardMarkup = github_kb_builder.as_markup(
    resize_keyboard=True)


# ------------------------ Клавиатура для выбора стиля ------------------------
# Создаём кнопку для преобразования фотографий в аниме:
to_anime_button = KeyboardButton(
    text=LEXICON_BUTTONS_RU['to_anime_button'])

# Создаём кнопку для преобразования аниме в фотографию:
to_selfie_button = KeyboardButton(
    text=LEXICON_BUTTONS_RU['to_selfie_button'])

# Создаём кнопку для преобразования фотографий в милое аниме:
to_cute_anime_button = KeyboardButton(
    text=LEXICON_BUTTONS_RU['to_cute_anime_button'])

# Иницализируем билдер для данной клавиатуры:
styles_kb_builder = ReplyKeyboardBuilder()

# Добавляем кнопки в билдер:
styles_kb_builder.row(
    to_anime_button, to_cute_anime_button,
    to_selfie_button, width=1)

# Создаём styles_keyboard клавиатуру:
styles_keyboard: ReplyKeyboardMarkup = styles_kb_builder.as_markup(
    resize_keyboard=True)
