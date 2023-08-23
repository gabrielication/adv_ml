import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        
        #CIFAR10 mean and std
        #self.mean = (0.4914, 0.4822, 0.4465)
        #self.std = (0.2471, 0.2435, 0.2616)
        
        #CIFAR100 mean and std
        self.mean = (0.5071, 0.4865, 0.4409)
        self.std = (0.2673, 0.2564, 0.2762)
        
    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
