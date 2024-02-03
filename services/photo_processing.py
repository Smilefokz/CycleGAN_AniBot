import os
import torchvision.transforms as tt

from PIL import Image
from torch.utils.data import Dataset

from config.model_config import IMAGE_SIZE


# Класс для формирования датасета SelfieToAnimeDataset:
class SelfieToAnimeDataset(Dataset):

    def __init__(self, directory: str) -> Dataset:

        path_list = os.listdir(directory)
        abspath = os.path.abspath(directory)

        self.directory = directory
        self.image_list = [os.path.join(abspath, path) for path in path_list]

        # Аугментации для преобразования полученных
        # от пользователя изображений:
        self.transform = tt.Compose([
            tt.Resize(IMAGE_SIZE),
            tt.CenterCrop(IMAGE_SIZE),
            tt.ToTensor(),
            tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, index: int):

        path = self.image_list[index]
        image = Image.open(path).convert('RGB')

        return self.transform(image)


# Аугментация для преобразования отправляемых пользователю изображений:
transform = tt.Compose([tt.Resize(IMAGE_SIZE * 2, antialias=True)])
