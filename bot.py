import asyncio
import logging

from aiogram import Bot, Dispatcher
from keyboards.main_menu import set_main_menu
from config.bot_config import Config, load_config
from handlers import user_handlers, other_handlers

# Инициализируем логгер:
logger = logging.getLogger(__name__)


async def main() -> None:
    """Функция конфигурирования и запуска бота."""

    # Конфигурируем логирование:
    logging.basicConfig(
        level=logging.INFO,
        format='[{asctime}] #{levelname:8} {filename}:'
               '{lineno} - {name} - {message}',
        style='{'
    )

    # Выводим информацию о запуске бота:
    logger.info("Starting Bot")

    # Загружаем конфигуратор:
    config: Config = load_config()

    # Инициализируем бота и диспетчер:
    bot = Bot(token=config.tg_bot.token,
              parse_mode='HTML')
    dp = Dispatcher()

    # Настраиваем главное меню бота:
    await set_main_menu(bot)

    # Регистрируем роутеры в диспетчере:
    dp.include_router(user_handlers.router)
    dp.include_router(other_handlers.router)

    # Пропускаем накопившиеся апдейты и запускаем polling:
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
