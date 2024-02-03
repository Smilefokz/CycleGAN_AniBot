from aiogram import Bot
from aiogram.types import BotCommand

from lexicon.lexicon_ru import LEXICON_COMMANDS_RU


async def set_main_menu(bot: Bot) -> None:
    """
    Функция для настройки Menu бота, использующая
    команды из словаря LEXICON_COMMANDS_RU.
    """
    commands = [BotCommand(command=command, description=description)
                for command, description in LEXICON_COMMANDS_RU.items()]

    await bot.set_my_commands(commands)
