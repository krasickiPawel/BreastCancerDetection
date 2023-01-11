from PIL import Image
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, paths, labels, trans=None) -> None:
        super(BreastDataset, self).__init__()
        self.paths = paths
        self.labels = labels
        self.transforms = trans

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_label = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_label


class BreastValDataset(Dataset):
    def __init__(self, paths, trans=None) -> None:
        super(BreastValDataset, self).__init__()
        self.paths = paths
        self.transforms = trans

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_path
