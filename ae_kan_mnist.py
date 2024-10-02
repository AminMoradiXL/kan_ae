from ae_kan_original import Autoencoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

size = 784

input_size = 28 * 28
hidden_size = size
bottleneck_size = size
model = Autoencoder(input_size, hidden_size, bottleneck_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

criterion = nn.MSELoss()

start_time = time.time()

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


rec_loss = val_loss

total_training_time = time.time() - start_time

print(f"Total Training Time: {total_training_time:.2f} seconds")

def visualize_reconstructions(model, dataloader, n_images=4):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.view(-1, input_size).to(device)
    
    with torch.no_grad():
        bottleneck = model.encoder(images)
        reconstructed = model.decoder(bottleneck)
    
    images = images.cpu().view(-1, 1, 28, 28)
    bottleneck = bottleneck.cpu().view(-1, bottleneck_size)  
    reconstructed = reconstructed.cpu().view(-1, 1, 28, 28)
    
    # Plot the images
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))
    for i in range(n_images):
        # Original 
        ax = axes[i, 0]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        ax.axis("off")
        # Bottleneck
        ax = axes[i, 1]
        ax.imshow(bottleneck[i].view(-1, 1).repeat(1, 3).numpy(), cmap='gray')
        ax.set_title("Bottleneck")
        ax.axis("off")
        # Reconstructed
        ax = axes[i, 2]
        ax.imshow(reconstructed[i].squeeze(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig('kan_mnist_recon.png', dpi = 600)
    plt.show()

print("Training set reconstructions:")
visualize_reconstructions(model, trainloader)

print("Validation set reconstructions:")
visualize_reconstructions(model, valloader)

# def evaluate_knn_on_reconstructed_images(model, dataloader):
#     model.eval()
#     reconstructed_images = []
#     labels = []
    
#     with torch.no_grad():
#         for images, targets in dataloader:
#             images = images.view(-1, input_size).to(device)
#             reconstructed = model(images)
#             reconstructed = reconstructed.cpu().view(-1, input_size)
#             reconstructed_images.append(reconstructed)
#             labels.append(targets)
    
#     reconstructed_images = torch.cat(reconstructed_images, dim=0).numpy()
#     labels = torch.cat(labels, dim=0).numpy()
    
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(reconstructed_images, labels)
    
#     predictions = knn.predict(reconstructed_images)
    
#     acc = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions, average='weighted')
#     cm = confusion_matrix(labels, predictions)
    
#     print("Confusion Matrix:\n", cm)
#     print("\nAccuracy: {:.4f}".format(acc))
#     print("F1 Score: {:.4f}".format(f1))
#     print("\nClassification Report:\n", classification_report(labels, predictions))
    
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.savefig('kan_mnist_conmat.png', dpi = 600)
#     plt.show()

# print("Evaluating KNN on reconstructed validation images:")
# evaluate_knn_on_reconstructed_images(model, valloader)

