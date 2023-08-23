import torch

from load_cifar100 import train_dataloader, val_dataloader, test_dataloader
from models.resnet import resnet18, resnet34, resnet50

from utils.utils import progress_bar

import os

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

def configure_optimizers(model, train_ds, lr=0.1, weight_decay=5e-4, max_epochs=100):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    return optimizer, scheduler

def train(epoch, model, train_loader, device, optimizer, criterion, progress_bar):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
def test(epoch, model, test_loader, device, criterion, progress_bar):
    global best_acc
    
    print("testing...")
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def train_resnet_on_cifar100(batch_size=256, num_workers=8):
    dev = set_gpu_device()
    
    train_ds = train_dataloader(batch_size, num_workers)
    val_ds = val_dataloader(batch_size, num_workers)
    test_ds = test_dataloader(batch_size, num_workers)
    
    model = resnet50()
    
    model.to(dev)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer, scheduler = configure_optimizers(model, train_ds)
    
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, model, train_ds, dev, optimizer, criterion, progress_bar)
        test(epoch, model, test_ds, dev, criterion, progress_bar)
        scheduler.step()

if __name__ == "__main__":
    
    train_resnet_on_cifar100()