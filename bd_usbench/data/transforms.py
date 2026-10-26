"""Image transforms. Light augmentation for train only; flips off by default
because laterality matters in abdominal/fetal ultrasound."""

from __future__ import annotations

from torchvision import transforms

from bd_usbench.utils import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    image_size: int,
    train: bool,
    hflip: bool = False,
    rotation_deg: float = 8.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    crop_scale: tuple[float, float] = (0.85, 1.0),
) -> transforms.Compose:
    """Build torchvision transforms for a PIL RGB image.

    Args:
        image_size: Side length of the square model input.
        train: If True, apply light augmentation (random crop, small rotation,
            brightness/contrast jitter, optional horizontal flip).
        hflip: Enable horizontal flip (default off — laterality matters).
        rotation_deg: Max absolute rotation in degrees for train augmentation.
        brightness: ColorJitter brightness factor.
        contrast: ColorJitter contrast factor.
        crop_scale: Area scale range for RandomResizedCrop.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    aug: list = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=(0.95, 1.05)),
        transforms.RandomRotation(rotation_deg, fill=0),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
    ]
    if hflip:
        aug.append(transforms.RandomHorizontalFlip(0.5))
    aug += [transforms.ToTensor(), normalize]
    return transforms.Compose(aug)
