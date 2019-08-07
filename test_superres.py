from src.metrics import PSNR
from src.learner import SuperResolutionLearner
from src.models import PixelShuffleDecoderV2

from nntoolbox.vision.utils import UnlabelledImageListDataset, UnsupervisedFromSupervisedDataset
from nntoolbox.callbacks import Tensorboard, LossLogger, \
    ModelCheckpoint, ToDeviceCallback
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor
from torchvision.datasets import CIFAR10
from torch.optim import Adam

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

print("Begin creating dataset")
images = UnlabelledImageListDataset("data/train2014/", transform=Compose(
    [
        Resize(512),
        RandomCrop((256, 256))
    ]
))

# images = UnsupervisedFromSupervisedDataset(CIFAR10(root="data/CIFAR/", download=True, train=True, transform=ToTensor()))

upscale_factor = 8
batch_size = 8

print("Begin splitting data")
train_size = int(0.80 * len(images))
val_size = len(images) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(images, [train_size, val_size])


print("Begin creating data dataloaders")
dataloader = DataLoader(train_dataset, batch_size=batch_size)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size)
# print(len(dataloader))

print("Creating models")

model = PixelShuffleDecoderV2()
print("Finish creating model")
optimizer = Adam(model.parameters())
learner = SuperResolutionLearner(
    dataloader, dataloader_val,
    model, criterion=nn.MSELoss(), optimizer=optimizer
)

metrics = {
    "psnr": PSNR()
}

callbacks = [
    ToDeviceCallback(),
    Tensorboard(),
    LossLogger(),
    # lr_scheduler,
    ModelCheckpoint(learner=learner, save_best_only=False, filepath='weights/model.pt'),
]
learner.learn(n_epoch=100, metrics=metrics, callbacks=callbacks, upscale_factor=upscale_factor, final_metric='psnr')
