import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np




class CIFAR10Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None: 
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Change input channels to 3 (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Adjust fully connected layers for 32x32 input
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from 32x32 input after conv + pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # Output: 6x28x28 → 6x14x14
        x = self.pool(F.relu(self.conv2(x)))  # Output: 16x10x10 → 16x5x5
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
class Net(nn.Module):
    def __init__(self, num_classes: int) -> None: 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
 
def train(net, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()

    total_loss = 0.0
    y_true, y_pred = [], []

    # Initialize gradient accumulator
    grad_sums = {
        name: torch.zeros_like(param, device=device)
        for name, param in net.named_parameters()
        if param.requires_grad
    }
    grad_steps = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate gradients
            for name, param in net.named_parameters():
                if param.grad is not None:
                    grad_sums[name] += param.grad.detach()
            grad_steps += 1

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(trainloader.dataset)

    # Average gradients
    avg_grads = {
        name: (grad / grad_steps).cpu().numpy()
        for name, grad in grad_sums.items()
    }

    return avg_loss, np.array(y_true), np.array(y_pred), avg_grads

    

def test(net, testloader, device: str):
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize variables for tracking loss and predictions
    correct, loss = 0, 0.0
    y_true, y_pred = [], []

    net.eval()
    net.to(device)
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            # Compute loss
            loss += criterion(outputs, labels).item()
            
            # Get predicted class labels
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Store true and predicted labels for metric calculations
            y_true.extend(labels.cpu().numpy())  
            y_pred.extend(predicted.cpu().numpy())  

    # Compute accuracy
    accuracy = correct / len(testloader.dataset)
    
    return loss, accuracy, np.array(y_true), np.array(y_pred)


def models_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters
    
    ndarray = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarray)
    return parameters