import os
import torch
from torchvision import datasets
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from .augmentation import train_augmentation, test_augmentation
import numpy as np
from torchvision import transforms


class CIFAR10_Filtered(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=False,
        classes=None,
        num_samples_per_class=None,
    ):
        super(CIFAR10_Filtered, self).__init__(
            root, train=train, transform=transform, download=download
        )

        if classes is not None:
            indices = self._filter_classes(classes)
            if num_samples_per_class is not None and train:
                indices = self._sample_indices(
                    indices, classes, num_samples_per_class)

            self.data = self.data[indices]
            self.targets = np.array(
                [classes.index(self.targets[i]) for i in indices])

    def _filter_classes(self, classes):
        # 返回只包含指定类别的样本的索引
        indices = [i for i, target in enumerate(
            self.targets) if target in classes]
        return indices

    def _sample_indices(self, indices, classes, num_samples_per_class):
        # 从每个类别中采样指定数量的样本
        sampled_indices = []
        class_counts = {cls: 0 for cls in classes}
        for i in indices:
            target = self.targets[i]
            if class_counts[target] < num_samples_per_class:
                sampled_indices.append(i)
                class_counts[target] += 1
        return sampled_indices


class CIFAR100_Filtered(datasets.CIFAR100):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=False,
        classes=None,
        num_samples_per_class=None,
    ):
        super(CIFAR100_Filtered, self).__init__(
            root, train=train, transform=transform, download=download
        )

        if classes is not None:
            indices = self._filter_classes(classes)
            if num_samples_per_class is not None and train:
                indices = self._sample_indices(
                    indices, classes, num_samples_per_class)

            self.data = self.data[indices]
            self.targets = np.array(
                [classes.index(self.targets[i]) for i in indices])

    def _filter_classes(self, classes):
        # 返回只包含指定类别的样本的索引
        indices = [i for i, target in enumerate(
            self.targets) if target in classes]
        return indices

    def _sample_indices(self, indices, classes, num_samples_per_class):
        # 从每个类别中采样指定数量的样本
        sampled_indices = []
        class_counts = {cls: 0 for cls in classes}
        for i in indices:
            target = self.targets[i]
            if class_counts[target] < num_samples_per_class:
                sampled_indices.append(i)
                class_counts[target] += 1
        return sampled_indices


class SVHN_Filtered(datasets.SVHN):
    def __init__(
        self,
        root,
        split,
        transform=None,
        download=False,
        classes=None,
        num_samples_per_class=None,
    ):
        super(SVHN_Filtered, self).__init__(
            root, transform=transform, download=download, split=split
        )

        if classes is not None:
            indices = self._filter_classes(classes)
            if num_samples_per_class is not None:
                indices = self._sample_indices(
                    indices, classes, num_samples_per_class)
            self.data = self.data[indices]
            # self.labels = np.array(self.labels)[indices].tolist()
            self.labels = np.array(
                [classes.index(self.labels[i]) for i in indices])

    def _filter_classes(self, classes):
        # 返回只包含指定类别的样本的索引
        indices = [i for i, label in enumerate(
            self.labels) if label in classes]
        return indices

    def _sample_indices(self, indices, classes, num_samples_per_class):
        # 从每个类别中采样指定数量的样本
        sampled_indices = []
        class_counts = {cls: 0 for cls in classes}
        for i in indices:
            label = self.labels[i]
            if class_counts[label] < num_samples_per_class:
                sampled_indices.append(i)
                class_counts[label] += 1
        return sampled_indices


def load_cifar10(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


def load_cifarplus10(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = CIFAR100_Filtered(
        root=os.path.join(datadir, "CIFAR100"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


def load_cifarplus50(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = CIFAR100_Filtered(
        root=os.path.join(datadir, "CIFAR100"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


def load_svhn(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = SVHN_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        split="train",
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = SVHN_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        split="train",
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = SVHN_Filtered(
        root=os.path.join(datadir, "SVHN"),
        split="test",
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


def load_cifar10_svhn(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = SVHN_Filtered(
        root=os.path.join(datadir, "SVHN"),
        split="test",
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


def load_cifar10_100(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = CIFAR100_Filtered(
        root=os.path.join(datadir, "CIFAR100"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )
    
    # OEset = DTD(
    #     root=os.path.join(datadir, "dtd"),
    #     transform=test_augmentation(img_size=img_size, mean=mean, std=std)
    # )

    return trainset, testset, openset

def load_cifar100_10(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = CIFAR100_Filtered(
        root=os.path.join(datadir, "CIFAR100"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR100_Filtered(
        root=os.path.join(datadir, "CIFAR100"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )
    
    # OEset = DTD(
    #     root=os.path.join(datadir, "dtd"),
    #     transform=test_augmentation(img_size=img_size, mean=mean, std=std)
    # )

    return trainset, testset, openset


class Tiny_ImageNet_Filtered(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        classes=None,
        num_samples_per_class=None,
        train=True,
    ):
        super(Tiny_ImageNet_Filtered, self).__init__(root, transform=transform)

        # 如果指定了类别，则过滤数据
        if classes is not None:
            # 直接操作 samples 和 targets（兼容 ImageFolder）
            new_samples = []
            new_targets = []
            for path, label in self.samples:
                if label in classes:
                    new_label = classes.index(label)
                    new_samples.append((path, new_label))
                    new_targets.append(new_label)

            self.samples = self.imgs = new_samples
            self.targets = new_targets

            # 可选：按 num_samples_per_class 采样
            if num_samples_per_class is not None and train:
                self._sample_by_class(num_samples_per_class, classes)

    def _sample_by_class(self, num_per_class, classes):
        class_counts = {cls: 0 for cls in range(len(classes))}
        new_samples = []
        new_targets = []
        for path, label in self.samples:
            if class_counts[label] < num_per_class:
                new_samples.append((path, label))
                new_targets.append(label)
                class_counts[label] += 1
        self.samples = self.imgs = new_samples
        self.targets = new_targets


def load_tinyimagenet(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    # known_classes = [192, 112, 145, 107, 91, 180, 144, 193,
    #  10, 125, 186, 28, 72, 124, 54, 77, 157, 169, 104, 166]
    # out_classes = list(set(list(range(0, 200)))-set(known_classes))
    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = Tiny_ImageNet_Filtered(
        root=os.path.join(datadir, "tiny-imagenet-200", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = Tiny_ImageNet_Filtered(
        root=os.path.join(datadir, "tiny-imagenet-200", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = Tiny_ImageNet_Filtered(
        root=os.path.join(datadir, "tiny-imagenet-200", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


class ImageNetResize(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target=10
    ):
        super(ImageNetResize, self).__init__(root, transform=transform)
        self.target = target
        self.samples = [(path, self.target) for path, _ in self.samples]
        self.targets = [self.target] * len(self.targets)

        def __getitem__(self, index):
            """
                返回格式：(image, target) 其中 target 固定为 self.target
                """
            image, _ = super().__getitem__(index)  # 忽略原始标签
            return image, self.target


def load_imgnr(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):
    num_samples_per_class = num_samples_per_class
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = ImageNetResize(
        root=os.path.join(datadir, "Imagenet_resize"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
    )

    return trainset, testset, openset


class ImageNetCrop(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target=10
    ):
        super(ImageNetCrop, self).__init__(root, transform=transform)
        self.target = target
        self.samples = [(path, self.target) for path, _ in self.samples]
        self.targets = [self.target] * len(self.targets)

        def __getitem__(self, index):
            """
                返回格式：(image, target) 其中 target 固定为 self.target
                """
            image, _ = super().__getitem__(index)  # 忽略原始标签
            return image, self.target


def load_imgnc(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):
    num_samples_per_class = num_samples_per_class
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = ImageNetCrop(
        root=os.path.join(datadir, "Imagenet"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
    )

    return trainset, testset, openset


class LSUNResize(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target=10
    ):
        super(LSUNResize, self).__init__(root, transform=transform)
        self.target = target
        self.samples = [(path, self.target) for path, _ in self.samples]
        self.targets = [self.target] * len(self.targets)

        def __getitem__(self, index):
            """
                返回格式：(image, target) 其中 target 固定为 self.target
                """
            image, _ = super().__getitem__(index)  # 忽略原始标签
            return image, self.target


def load_lsunr(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):
    num_samples_per_class = num_samples_per_class
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = LSUNResize(
        root=os.path.join(datadir, "LSUN_resize"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
    )

    return trainset, testset, openset


def load_lsunc(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):
    num_samples_per_class = num_samples_per_class
    trainset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=True,
        download=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = CIFAR10_Filtered(
        root=os.path.join(datadir, "CIFAR10"),
        train=False,
        download=True,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = LSUNResize(
        root=os.path.join(datadir, "LSUN"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
    )

    return trainset, testset, openset


class ImageNet30_Filtered(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        classes=None,
        num_samples_per_class=None,
        train=True,
    ):
        super(ImageNet30_Filtered, self).__init__(root, transform=transform)

        self.full_classes = [
            'acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo',
            'barn', 'bikini', 'digital_clock', 'dragonfly', 'dumbbell',
            'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass',
            'manhole_cover', 'mosque', 'nail', 'parking_meter', 'pillow',
            'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
            'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano'
        ]

        if classes is None:
            classes = list(range(len(self.full_classes)))

        selected_classes = [self.full_classes[idx] for idx in classes]
        self.selected_indices = classes

        # 过滤样本
        new_samples = []
        new_targets = []
        for path, original_label in self.samples:
            original_class = self.classes[original_label]  # 原始类别名
            if original_class in selected_classes:
                # 将标签映射到新索引（保持与classes参数顺序一致）
                new_label = selected_classes.index(original_class)
                new_samples.append((path, new_label))
                new_targets.append(new_label)

        self.samples = self.imgs = new_samples
        self.targets = new_targets
        self.classes = selected_classes  # 更新为实际选中的类别名

        # 采样控制
        if num_samples_per_class is not None and train:
            self._sample_by_class(num_samples_per_class)

    def _sample_by_class(self, num_per_class):
        class_counts = {label: 0 for label in set(self.targets)}
        new_samples = []
        new_targets = []

        for path, label in self.samples:
            if class_counts[label] < num_per_class:
                new_samples.append((path, label))
                new_targets.append(label)
                class_counts[label] += 1

        self.samples = self.imgs = new_samples
        self.targets = new_targets


def load_imagenet30(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):
    num_samples_per_class = num_samples_per_class
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = ImageNet30_Filtered(
        root=os.path.join(datadir, "imagenet30", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = ImageNet30_Filtered(
        root=os.path.join(datadir, "imagenet30", "test"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = ImageNet30_Filtered(
        root=os.path.join(datadir, "imagenet30", "test"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


class ImageNet_Filtered(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        classes=None,
        num_samples_per_class=None,
        train=True,
    ):
        super(ImageNet_Filtered, self).__init__(root, transform=transform)

        # 如果指定了类别，则过滤数据
        if classes is not None:
            indices = self._filter_classes(classes)
            if num_samples_per_class is not None and train:
                indices = self._sample_indices(
                    indices, classes, num_samples_per_class)
            self.samples = [self.samples[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]
            self.imgs = self.samples  # 保持一致性

    def _filter_classes(self, classes):
        # 返回只包含指定类别的样本的索引
        indices = [i for i, label in enumerate(
            self.targets) if label in classes]
        return indices

    def _sample_indices(self, indices, classes, num_samples_per_class):
        # 从每个类别中采样指定数量的样本
        sampled_indices = []
        class_counts = {cls: 0 for cls in classes}
        for i in indices:
            label = self.targets[i]
            if class_counts[label] < num_samples_per_class:
                sampled_indices.append(i)
                class_counts[label] += 1
        return sampled_indices


def load_imagenet1k(datadir: str, img_size: int, mean: tuple, std: tuple, num_samples_per_class, known, unknown):

    # known_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    #                  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    # out_classes = list(set(list(range(0, 1000)))-set(known_classes))
    num_samples_per_class = 16
    # 加载训练集，只包含已知类别，并采样每个类别 num_samples_per_class 个样本
    trainset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载开放集，只包含未知类别
    openset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=unknown,
    )

    return trainset, testset, openset


class iNaturalist(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target=1000
    ):
        super(iNaturalist, self).__init__(root, transform=transform)
        self.target = target
        # 修改samples和targets，将所有样本标签统一为target
        self.samples = [(path, self.target) for path, _ in self.samples]
        self.targets = [self.target] * len(self.targets)
        self.imgs = self.samples  # 保持一致性（可选）

class DTD(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target=10
    ):
        image_dir = os.path.join(root, "images")
        # 调用父类构造函数以加载图像
        super().__init__(root=image_dir, transform=transform)
        # 将所有样本的标签替换为固定的target值
        self.targets = [target] * len(self.targets)
        

class Textures_OOD(ImageFolder):
    """
    Textures数据集作为OOD数据集
    路径结构：~/nas-resource-linkdata/Textures/dtd/images/
    包含多个类别子文件夹：banded, blotchy, braided, 等等
    """
    def __init__(
        self,
        root,
        transform=None,
        target=10  # 统一标签，表示OOD样本
    ):
        super(Textures_OOD, self).__init__(root, transform=transform)
        
        # 将所有样本的标签统一设置为target（表示OOD）
        self.samples = [(path, target) for path, _ in self.samples]
        self.targets = [target] * len(self.targets)
        self.imgs = self.samples  # 保持一致性
        
        # 打印统计信息
        print(f"Loaded Textures OOD dataset:")
        print(f"  - Root directory: {root}")
        print(f"  - Number of categories: {len(self.classes)}")
        print(f"  - Total samples: {len(self)}")
        print(f"  - Sample categories: {self.classes[:5]}...")  # 显示前5个类别
        

def load_imgn1k_textures(
    datadir: str, 
    img_size: int, 
    mean: tuple, 
    std: tuple, 
    num_samples_per_class: int, 
    known: list, 
    unknown: list = None
):
    """
    加载ImageNet-1k作为ID，Textures作为OOD
    """
    
    num_samples_per_class = num_samples_per_class
    
    # 加载ImageNet-1k训练集（ID数据）
    trainset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载ImageNet-1k验证集（ID测试数据）
    testset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载Textures作为OOD数据集
    textures_root = os.path.join(datadir, "Textures", "dtd", "images")
    
    openset = Textures_OOD(
        root=textures_root,
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        target=1000  # 统一OOD标签
    )
    
    return trainset, testset, openset
        

def load_imgn1k_inaturalist(
    datadir: str, 
    img_size: int, 
    mean: tuple, 
    std: tuple, 
    num_samples_per_class: int, 
    known: list, 
    unknown: list = None  # 这里unknown参数不使用，因为OOD是独立数据集
):
    """
    加载ImageNet-1k作为ID，iNaturalist作为OOD
    
    参数:
        datadir: 数据集根目录
        img_size: 图像尺寸
        mean: 归一化均值
        std: 归一化标准差
        num_samples_per_class: 每个类别的训练样本数（few-shot设置）
        known: ImageNet-1k的类别列表（应为0-999的列表）
        unknown: 此参数不使用，仅保持接口一致
    """
    
    # 根据文章中的few-shot设置，每个类别采样指定数量的样本
    # 文章中使用的是4-shot（每个类别4个样本）或16-shot
    num_samples_per_class = num_samples_per_class
    
    # 加载ImageNet-1k训练集（ID数据）
    trainset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载测试集，只包含已知类别
    testset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载iNaturalist作为OOD数据集
    # 注意：文章中提到需要确保iNaturalist中的类别不与ImageNet-1k重叠
    # 通常会手动选择或过滤掉重叠类别
    openset = iNaturalist(
        root=os.path.join(datadir, "iNaturalist"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        target=1000  # 统一OOD标签，与ID标签（0-999）区分开
    )

    return trainset, testset, openset


def load_imgn1k_sun(
    datadir: str, 
    img_size: int, 
    mean: tuple, 
    std: tuple, 
    num_samples_per_class: int, 
    known: list, 
    unknown: list = None
):
    """
    加载ImageNet-1k作为ID，SUN作为OOD
    直接复用iNaturalist类加载SUN数据集
    """
    
    num_samples_per_class = num_samples_per_class
    
    # 加载ImageNet-1k训练集（ID数据）
    trainset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载ImageNet-1k验证集（ID测试数据）
    testset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "val"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载SUN作为OOD数据集，复用iNaturalist类
    openset = iNaturalist(  # 或 iNaturalist_OOD，取决于你实际定义的类名
        root=os.path.join(datadir, "SUN"),  # 只改这个路径
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        target=1000  # 统一OOD标签
    )
    
    print(f"Loaded SUN OOD dataset with {len(openset)} samples")
    
    return trainset, testset, openset

def load_imgn1k_places(
    datadir: str, 
    img_size: int, 
    mean: tuple, 
    std: tuple, 
    num_samples_per_class: int, 
    known: list, 
    unknown: list = None
):
    """
    加载ImageNet-1k作为ID，Places作为OOD
    """
    
    num_samples_per_class = num_samples_per_class
    
    # 加载ImageNet-1k训练集（ID数据）
    trainset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        train=True,
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
        num_samples_per_class=num_samples_per_class,
    )

    # 加载ImageNet-1k验证集（ID测试数据）
    testset = ImageNet_Filtered(
        root=os.path.join(datadir, "ImageNet1k", "train"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        classes=known,
    )

    # 加载Places作为OOD数据集，复用iNaturalist类
    openset = iNaturalist(  # 或 iNaturalist_OOD，取决于你实际定义的类名
        root=os.path.join(datadir, "Places"),  # 只改这个路径
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        target=1000  # 统一OOD标签
    )
    
    print(f"Loaded Places OOD dataset with {len(openset)} samples")
    
    return trainset, testset, openset  


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16
    )
