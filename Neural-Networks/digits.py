import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

BATCHSIZE = 100
LR1 = 0.1
LR2 = 0.1

#! DATASETS
#? 3d tensors of 60,000 stacked 28x28 matrices | Represent 60k examples of 28x28 pixel values
# Values are in grayscale, so 0-255
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

def visualize(n: int, test: bool = False):
    """Show the example N from the datasets"""
    dig = {0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}
    data = test_data if test else train_data
    plt.title(dig[data.targets[n].item()])
    plt.imshow(data.data[n], cmap='gray')


#! MODELS
#? 90% accuracy after 30s of training
class BasicModel(nn.Module):
    def __init__(self, outputs: int = 10):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_features=outputs))
        self.optimizer = optim.SGD(self.parameters(), lr=LR1)

    def forward(self, x):
        return self.layers(x).softmax(-1)
    
    def trains(self, epochs: int = 5):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for imgs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(imgs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    @torch.no_grad()
    def test(self):
        self.eval()
        correct = 0
        total = 0
        for imgs, labels in test_loader:
            outputs = self(imgs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        print(f"Test Accuracy: {correct / total * 100:.2f}%")
    
    @torch.no_grad()
    def predict(self, img):
        """model.predict(test_data.data[n].float())"""
        self.eval()
        with torch.no_grad():
            output = self(img.unsqueeze(0))
            return output.argmax().item()

