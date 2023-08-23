import torch

from load_cifar100 import train_dataloader, val_dataloader, test_dataloader
from models.resnet import resnet18, resnet34, resnet50

def set_gpu_device():
    
    if torch.backends.mps.is_available():
        print("Apple M1 MPS available")
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA available")
        dev = torch.device("cuda")
    else:
        print ("No GPU found. CPU only")
        dev = torch.device("cpu")
        
    return dev

def train_resnet_on_cifar100(batch_size=256, num_workers=8):
    dev = set_gpu_device()
    
    train_ds = train_dataloader(batch_size, num_workers)
    val_ds = val_dataloader(batch_size, num_workers)
    test_ds = test_dataloader(batch_size, num_workers)
    
    model = resnet50()
    
    return None

if __name__ == "__main__":
    
    train_resnet_on_cifar100()