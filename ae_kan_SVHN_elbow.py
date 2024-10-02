from ae_kan import Autoencoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import time

# Transform to normalize the SVHN images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the SVHN dataset
trainset = torchvision.datasets.SVHN(
    root="./data", split='train', download=True, transform=transform
)
valset = torchvision.datasets.SVHN(
    root="./data", split='test', download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)


import pandas as pd

rec = pd.DataFrame(columns=['size', 'loss'])

for k in range(0, 3072, 50):

    size = 3072 - k

    input_size = 32 * 32 * 3  # Updated input size for 32x32 RGB images
    hidden_size = size  # Adjust these sizes as needed
    bottleneck_size = size
    model = Autoencoder(input_size, hidden_size, bottleneck_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, _) in enumerate(pbar):
                images = images.view(-1, input_size).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, images)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in valloader:
                images = images.view(-1, input_size).to(device)
                output = model(images)
                val_loss += criterion(output, images).item()
        val_loss /= len(valloader)
    
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")
    
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        scheduler.step()
    
    print(f'size = {size}, loss = {val_loss}')
    rec.loc[k, 'size'] = size
    rec.loc[k, 'loss'] = val_loss
    
rec = rec.sort_values('size', ascending=True)
rec.plot( x = 'size', y = 'loss')
