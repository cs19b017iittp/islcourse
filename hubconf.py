import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

class cs19b017NN(nn.Module):
  def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(train_data_loader=None, n_epochs=10):
    model = cs19b017NN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for i, (images, labels) in enumerate(train_data_loader):
      # origin shape: [4, 3, 32, 32] = 4, 3, 1024
      # input_layer: 3 input channels, 6 output channels, 5 kernel size
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = loss_fn(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
      print ('Returning model... (rollnumber: xx)')
  
      return model
