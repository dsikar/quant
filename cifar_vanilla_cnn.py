import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pickle
from datetime import datetime

date_string = datetime.now().strftime("%Y%m%d%H%M%S_")

start_time = time.time()
print("Start time:", time.strftime("%H:%M:%S", time.localtime(start_time)))

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, 1)  # Change input channels to 3
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(12544, 128)  # Change input size to match CIFAR10 dimensions
#         self.fc2 = nn.Linear(128, 10)  # Change output size to 10

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1) # input 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # input 30x30
        self.conv3 = nn.Conv2d(64, 128, 3, 1) # input 28x28
        self.conv4 = nn.Conv2d(128, 256, 3, 1) # input 26x26
        self.conv5 = nn.Conv2d(256, 512, 3, 1) # input 24x24
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)   # 3x3x512 = 4608
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



# Loading and normalizing MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load training data
trainset = CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Load test data
testset = CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)


# Instantiating the model
net = Net()

# Defining the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Placeholder for the losses and accuracies
train_losses = []
train_accuracies = []

# Training the network and tracking the loss and accuracy
for epoch in range(70):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # track loss
        running_loss += loss.item()

        # track accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

# Plotting the training loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, '-o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, '-o')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# Save the model to a file
model_path = f"./{date_string}_cifar_cnn.pth"
torch.save(net.state_dict(), model_path)
print("Model has been saved to {}".format(model_path))

# Save losses and accuracies, prepend date_string to filename
file_name = f"{date_string}_cifar_training.pkl"
with open(file_name, 'wb') as f:
    pickle.dump((train_losses, train_accuracies), f)
# Print the name of the saved file
print(f"File saved: {file_name}")

# 3. Print the system time and the running time since step 1
end_time = time.time()
elapsed_time = end_time - start_time
print("End time:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print("Elapsed time (hh:mm:ss):", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

