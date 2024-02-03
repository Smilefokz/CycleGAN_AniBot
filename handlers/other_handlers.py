import os
from aiogram import Router
from aiogram.types import Message

from config.model_config import IN_PATH
from lexicon.lexicon_ru import LEXICON_RU

router = Router()


# Данный хэндлер предназначен для сообщений,
# которые не попали в другие хэндлеры:
@router.message()
async def send_message(message: Message) -> None:

    folder = message.from_user.id

    if len(os.listdir(f'{IN_PATH}/{folder}')) > 0:
        await message.answer(text=LEXICON_RU['style'])
    else:
        await message.answer(text=LEXICON_RU['answer'])
