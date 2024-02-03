from environs import Env
from dataclasses import dataclass


@dataclass
class TgBot:
    token: str      # Токен для доступа к телеграм-боту


@dataclass
class Config:
    tg_bot: TgBot


def load_config(path: str | None = None) -> Config:
    """
    Функция  считывает данные из файла .env и возвращает
    экземпляр класса Config с заполненным полем token.

    Параметры:
    path - путь до файла .env (по умолчанию None, если файл находится
    в одной директории с файлом bot).
    """
    env = Env()
    env.read_env(path)

    return Config(tg_bot=TgBot(token=env('BOT_TOKEN')))
