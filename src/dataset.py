import os

from PIL import Image
from torch.utils.data import Dataset


class AmazonDataset(Dataset):
    def __init__(self, df, ohe_tags, transform, path):
        super().__init__()
        self.df = df
        self.ohe_tags = ohe_tags
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx].image_name + ".jpg"
        if filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
        else:
            raise Exception(f"Can't fetch {filename} among {self.paths}")
        img = Image.open(file_path).convert("RGB")
        label = self.ohe_tags[idx]
        img = self.transform(img)
        return img, label.astype("float32")
