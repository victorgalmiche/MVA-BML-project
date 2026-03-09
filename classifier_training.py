import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Loading the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

# Creating the classifier model 
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),   # (1, 28, 28) -> (32, 28, 28)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),  # (32, 28, 28) -> (64, 28, 28)
            nn.MaxPool2d(2),                             # (64, 28, 28) -> (64, 14, 14)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), # (64, 14, 14) -> (128, 14, 14)
            nn.MaxPool2d(2),                             # (128, 14, 14) -> (128, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (128, 7, 7) -> (128*7*7)
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),      # (128*7*) -> (256)
            nn.Dropout(0.5),
            nn.Linear(256, 10)                           # (256) -> (10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# We train on the whole MNIST dataset
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model = MnistCNN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss() # The loss is the cross-entropy

for epoch in range(5): # We train for 5 epochs
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5 done")


# And save the model
torch.save(model.state_dict(), "./models/mnist_cnn.pth")
