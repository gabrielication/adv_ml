# Importing all the essential libraries

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set the device

if torch.backends.mps.is_available():
    print("Apple M1 MPS available")
    dev = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA available")
    dev = torch.device("cuda")
else:
    print ("No GPU found. CPU only")
    dev = torch.device("cpu")
    
x = torch.ones(1, device=dev)
print(x)

# Initializing the required hyperparameters

num_classes = 10
input_size = 784
batch_size = 64
lr = 0.0001
epochs = 3

# Collection of data

'''
Here, a transformation pipeline is being defined using 
torchvision.transforms.Compose. In this specific pipeline, only one 
transformation is applied: torchvision.transforms.ToTensor(). This 
transformation converts PIL images (the format in which the MNIST dataset 
is originally loaded) to PyTorch tensors and also scales the pixel values 
from [0, 255] to [0.0, 1.0].
'''

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

'''
    root='/datasets': Specifies the directory where the dataset will be stored. If the dataset is not present in this directory, it will be downloaded.
    train=True: Indicates that the training portion of the MNIST dataset should be loaded.
    download=True: If the dataset isn't found in the specified root directory, it will be downloaded.
    transform=T: Applies the previously defined transformation (conversion to tensor) to the data.
'''

X_train = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=T)

'''
The DataLoader class in PyTorch is used to create mini-batches of data, shuffle them (if required), and provide an interface to iterate over these batches during training.

    dataset=X_train: Specifies the dataset from which the data should be loaded.
    batch_size=batch_size: Defines the size of each mini-batch.
    shuffle=True: This will shuffle the dataset before creating batches, which is generally a good practice during training to ensure the model doesn't memorize the order of samples.
'''
train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)

'''
This is similar to the line for loading the training data but with train=False, 
indicating that the testing (or validation) portion of the MNIST dataset should 
be loaded.
'''
X_test = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=T)
test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=True)

# Constructing the model

class neural_network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
model = neural_network(input_size=input_size, num_classes=num_classes)
    
# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the network

model.to(dev)

for epoch in range(epochs):
    for batch, (data, target) in enumerate(train_loader):
        # Obtaining the cuda parameters
        data = data.to(device=dev)
        target = target.to(device=dev)

        # Reshaping to suit our model
        data = data.reshape(data.shape[0], -1)

        # Forward propagation
        score = model(data)
        loss = criterion(score, target)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Check the performance

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=dev)
            y = y.to(device=dev)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        if num_samples == 60000:
            print(f"Train accuracy = "
                  f"{float(num_correct) / float(num_samples) * 100:.2f}")
        else:
            print(f"Test accuracy = "
                  f"{float(num_correct) / float(num_samples) * 100:.2f}")

    # model.eval() and model.train() are just flags, they set
    # the behavior of some layer like dropout or others
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)