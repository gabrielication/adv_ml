from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

#CIFAR10 mean and std
#self.mean = (0.4914, 0.4822, 0.4465)
#self.std = (0.2471, 0.2435, 0.2616)
        
#CIFAR100 mean and std
mean = (0.5071, 0.4865, 0.4409)
std = (0.2673, 0.2564, 0.2762)

def train_dataloader(batch_size=256, num_workers=8):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        dataset = CIFAR100(root=".", train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
def val_dataloader(batch_size=256, num_workers=8):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        dataset = CIFAR100(root=".", train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
def test_dataloader(batch_size=256, num_workers=8):
        return val_dataloader(batch_size, num_workers)