from tqdm import tqdm
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

num_epochs = 10


# size 3*3

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Encoder layers
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # (N, 1, 28, 28) -> (N, 16, 14, 14)
#             nn.ReLU(),
#             nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # (N, 16, 14, 14) -> (N, 32, 7, 7)
#             nn.ReLU(),
#             nn.Conv2d(4, 1, kernel_size=3, stride=3, padding=1)   # (N, 32, 7, 7) -> (N, 64, 4, 4)
#         )
        
#         # Decoder layers
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1, 4, kernel_size=3, stride=3, padding=1, output_padding=1),  # (N, 64, 4, 4) -> (N, 32, 7, 7)
#             nn.ReLU(),
#             nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 32, 7, 7) -> (N, 16, 14, 14)
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 16, 14, 14) -> (N, 1, 28, 28)
#             nn.Tanh()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
   
# size 6*3
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # (N, 1, 28, 28) -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # (N, 16, 14, 14) -> (N, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, stride=3, padding=1)   # (N, 32, 7, 7) -> (N, 64, 4, 4)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 4, kernel_size=3, stride=3, padding=1, output_padding=1),  # (N, 64, 4, 4) -> (N, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 32, 7, 7) -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (N, 16, 14, 14) -> (N, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = MNIST(root="./data", train=True, download=True, transform=transform)
valset = MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

start_time = time.time()


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    # Training loop with tqdm progress bar
    with tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            encoded, outputs = model(images)
            
            # Ensure outputs and images have the same shape
            if outputs.shape != images.shape:
                outputs = outputs[:, :, :28, :28]  # Trim output to match input size
            
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            # Update tqdm progress bar
            pbar.set_postfix(loss=train_loss / len(trainloader))
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in valloader:
            images = images.to(device)
            encoded, outputs = model(images)
            
            if outputs.shape != images.shape:
                outputs = outputs[:, :, :28, :28]
            
            val_loss += criterion(outputs, images).item() * images.size(0)
    
    val_loss /= len(valloader.dataset)
    
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler.step()

total_training_time = time.time() - start_time

print(f"Total Training Time: {total_training_time:.2f} seconds")

import matplotlib.pyplot as plt
import torch

def visualize_reconstructions(model, dataloader, n_images=3):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        encoded, reconstructed = model(images)
    
    images = images.cpu().numpy()
    encoded = encoded.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Plot the images
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))
    for i in range(n_images):
        # Original 
        ax = axes[i, 0]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        ax.axis("off")
        # Bottleneck (Encoded)
        ax = axes[i, 1]
        ax.imshow(encoded[i].reshape(encoded[i].shape[1], -1), cmap='gray')
        ax.set_title("Bottleneck")
        ax.axis("off")
        # Reconstructed
        ax = axes[i, 2]
        ax.imshow(reconstructed[i].squeeze(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# Example usage:
print("Training set reconstructions:")
visualize_reconstructions(model, trainloader)

print("Validation set reconstructions:")
visualize_reconstructions(model, valloader)

def evaluate_knn_on_encoded_features(model, dataloader):
    model.eval()
    encoded_features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            encoded, _ = model(images)
            
            encoded_features.append(encoded.view(encoded.size(0), -1).cpu()) 
            labels.append(targets.cpu())
    
    encoded_features = torch.cat(encoded_features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(encoded_features, labels)
    
    predictions = knn.predict(encoded_features)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)
    
    print("Confusion Matrix:\n", cm)
    print("\nAccuracy: {:.4f}".format(acc))
    print("F1 Score: {:.4f}".format(f1))
    print("\nClassification Report:\n", classification_report(labels, predictions))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
print("Evaluating KNN on encoded features of validation images:")
evaluate_knn_on_encoded_features(model, valloader)
