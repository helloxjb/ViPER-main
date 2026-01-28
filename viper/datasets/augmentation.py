from torchvision import transforms
from torchvision.transforms import RandAugment

def train_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def test_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform
