from matplotlib import pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.dataset import PatchifyTransform, WayangKulit
from src.models.basic import ViT
from train import PATCH_SIZE, MODELS_DIR, BASE_DIR

if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.63528919, 0.57810118, 0.51988552), (0.33020571, 0.34510824, 0.36673283)),
            PatchifyTransform(PATCH_SIZE)
        ]
    )
    ds = WayangKulit('dataset/val', transform=None)

    model = ViT.load_from_checkpoint(MODELS_DIR.joinpath('epoch=91-step=552.ckpt'))
    model.eval()

    im, c = ds[120]

    inp = transform(im).unsqueeze(0)
    res = model(inp)
    res = res.argmax().item()

    print(f"Predicted class: {res} - {ds.classes[res]}")
    print(f"Target class: {c} - {ds.classes[c]}")

    plt.imshow(im)
    plt.show()
