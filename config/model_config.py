# Размер изображения, подающийся на вход моделям:
IMAGE_SIZE = 128

# Пути для получаемых и отправляемых изображений:
IN_PATH = 'photo_storage/in'
OUT_PATH = 'photo_storage/out'

# URL для скачивания фотографий и аниме изображений:
SELFIE_URL = 'https://disk.yandex.ru/d/coNZQBOBhA_eUw'
ANIME_URL = 'https://disk.yandex.ru/d/Pcwinm8DA6wJ3Q'
CUTE_ANIME_URL = 'https://disk.yandex.ru/d/JOkPJ1PTANLRcQ'

# URL для перехода на страницу проекта в GitHub:
GIT_URL = 'https://github.com/Smilefokz/CycleGAN_AniBot'

# Пути для хранения весов моделей:
SELFIE_PATH = 'services/weights/to_selfie'
ANIME_PATH = 'services/weights/to_anime'
CUTE_ANIME_PATH = 'services/weights/to_cute_anime'

# Пути для хранения весов "из аниме в фотографию":
TO_SELFIE_D_ANIME = f'{SELFIE_PATH}/D_anime'
TO_SELFIE_D_SELFIE = f'{SELFIE_PATH}/D_selfie'
TO_SELFIE_G_ANIME_TO_SELFIE = f'{SELFIE_PATH}/G_anime_to_selfie'
TO_SELFIE_G_SELFIE_TO_ANIME = f'{SELFIE_PATH}/G_selfie_to_anime'

# Пути для хранения весов "из фотографии в аниме":
TO_ANIME_D_ANIME = f'{ANIME_PATH}/D_anime'
TO_ANIME_D_SELFIE = f'{ANIME_PATH}/D_selfie'
TO_ANIME_G_ANIME_TO_SELFIE = f'{ANIME_PATH}/G_anime_to_selfie'
TO_ANIME_G_SELFIE_TO_ANIME = f'{ANIME_PATH}/G_selfie_to_anime'

# Пути для хранения весов "из фотографии в милое аниме":
TO_CUTE_ANIME_D_ANIME = f'{CUTE_ANIME_PATH}/D_anime'
TO_CUTE_ANIME_D_SELFIE = f'{CUTE_ANIME_PATH}/D_selfie'
TO_CUTE_ANIME_G_ANIME_TO_SELFIE = f'{CUTE_ANIME_PATH}/G_anime_to_selfie'
TO_CUTE_ANIME_G_SELFIE_TO_ANIME = f'{CUTE_ANIME_PATH}/G_selfie_to_anime'
