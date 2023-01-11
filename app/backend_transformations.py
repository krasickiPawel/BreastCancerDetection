import imghdr

import torchvision.transforms as transforms
from PIL import Image


def get_transform_train(input_required_size=224):
    return transforms.Compose([
        transforms.Resize((input_required_size, input_required_size)),
        transforms.CenterCrop(input_required_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_transform_test(input_required_size=224):
    return transforms.Compose([
        transforms.Resize((input_required_size, input_required_size)),
        transforms.CenterCrop(input_required_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_test_trans(input_required_size=224):
    return (
        get_transform_train(input_required_size),
        get_transform_test(input_required_size)
    )


def transform_singe_image(img_path, input_required_size=224):
    if imghdr.what(img_path) is not None:
        img = Image.open(img_path).convert('RGB')
        return get_transform_test(input_required_size)(img).unsqueeze(0)
