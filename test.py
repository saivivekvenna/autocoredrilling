import torch                    # Core PyTorch library
import torch.nn as nn            # For building neural network layers
import torch.optim as optim      # For optimization algorithms (like SGD or Adam)
import torchvision               # For datasets, models, and image transforms
import torchvision.transforms as transforms  # For image preprocessing
from torch.utils.data import DataLoader      # To load data in batches

#need to check where to run compute 
device = (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Device Running on {device}")


#need to transform images from the start to the end 
transform = transforms.Compose([
    transforms.Resize((64, 64)),          # Resize all images to 64x64 pixels
    transforms.ToTensor(),                # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values (-1 to 1)
])

#Load a sample dataset (CIFAR10 has 10 classes of small 32x32 color images)
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# batch loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# define a simple cnn to look at images 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolution layer: 3 input channels (RGB), 16 output filters
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Second convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Pooling layer to reduce image size
        self.pool = nn.MaxPool2d(2, 2)
        # Flattened image size = 32 filters * (64/4 * 64/4) pixels = 32 * 16 * 16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for CIFAR10

# reminder: relu is where you take all the neg values and make them 0. 
# pooling is where you take the maxiumum(or average) from each patch inorder to reduce the amount of features. 

    def forward(self, x):
        # Apply first conv layer + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second conv layer + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the image for the fully connected layer
        x = x.view(-1, 32 * 16 * 16)
        # Pass through first fully connected layer + ReLU
        x = torch.relu(self.fc1(x))
        # Final layer (no activation, handled by loss)
        x = self.fc2(x)
        return x
    
# init the model 
model = SimpleCNN().to(device)

#cross entropy loss is a mix of soft max and negative log liklihood(idk); good at classifiying for multiclass
criterion = nn.CrossEntropyLoss()          

#adam is an adaptive optimizer, a smart version of gd 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔁 Training loop
for epoch in range(3):  # Run for 3 epochs (passes through entire dataset)
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 1️⃣ Reset gradients from previous step
        optimizer.zero_grad()

        # 2️⃣ Forward pass (get model predictions)
        outputs = model(images)

        # 3️⃣ Calculate loss
        loss = criterion(outputs, labels)

        # 4️⃣ Backward pass (calculate gradients)
        loss.backward()

        # 5️⃣ Update weights
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/3] - Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training complete!")



