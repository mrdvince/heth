import fire
import numpy as np
import rasterio as rio
import torch
from torchvision import transforms

from train import Model


def load(model_path):
    model = Model.load_from_checkpoint(model_path)
    model.eval()
    return model.to("cpu")


def inference(model_path, tiff_path):
    img = rio.open(tiff_path)
    img = np.float32(img.read())
    img = torch.from_numpy(img)
    img = transforms.ToPILImage()(img)
    img = transforms.Resize((224,))(img)
    img = transforms.ToTensor()(img)

    model = load(model_path)
    with torch.no_grad():
        y_hat = model(img.unsqueeze(0))
        y_hat = y_hat.detach().float().cpu().numpy()

    return (y_hat > 0.75).astype(float)


if __name__ == "__main__":
    fire.Fire(inference)
