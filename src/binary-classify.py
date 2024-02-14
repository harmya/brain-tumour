import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

print("------------Check if CUDA is available-------------")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

'''
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Device:", device)
'''

print("------------Important Variables----------------")
batch_size = 32
learning_rate = 0.005
epochs = 20
kernel_size = 5
print("Batch Size:", batch_size)
print("Learning Rate:", learning_rate)
print("Epochs:", epochs)
print("Kernel Size:", kernel_size)

print("--------------Loading Data----------------------")
print(".")
print(".")
print(".")
training = torch.load('torch-dataset/training.pt')
# training = training[:256]
validation = torch.load('torch-dataset/validation.pt')
testing = torch.load('torch-dataset/testing.pt')
# testing = testing[:256]

d_train = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
d_vali = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)
d_test = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=False)

print(type(d_train))

print("----------------- Data Loaded ---------------------\n")
print("Number of training samples: ")
print(len(d_train.dataset))
print("Number of validation samples: ")
print(len(d_vali.dataset))
print("Number of testing samples: ")
print(len(d_test.dataset))
print("--------------------------------------------------")


print("--------------Model Architecture------------------")
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=kernel_size, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=1),

    nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=1),

    nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=kernel_size, stride=1),

    nn.Flatten(),
    nn.Linear(128 * 25 * 25, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# check the output size
print("-------------------Checking input-output compatibility-------------------")
print("Input shape: ", training[0][0].shape)
x = training[0][0].unsqueeze(0)
out = model(x)
print("Output shape: ", out.shape)
print("Output: ", out.flatten().item())

model.to(device)
print(model)
print("-------------------------------------------------\n")


print("--------------Loss Function and Optimizer---------")
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print("Loss Function:", loss_function)
print("Optimizer:", optimizer)
print("-------------------------------------------------\n")


print("--------------Training Loop------------------------")
# training loop
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for image, label in d_train:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output.flatten(), label.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        for image, label in d_vali:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = loss_function(output.flatten(), label.float())
            val_loss += loss.item()
            output = (output > 0.5).flatten().int()
            correct += (output == label).sum().item()
            total += output.size(0)
        
        print(f"Epoch {epoch+1}, training loss: {running_loss / len(d_train)}, val_loss: {val_loss / len(d_vali)}, val_acc: {correct / total}")


print("-------------------------------------------------\n")


print("--------------Testing Loop------------------------")
correct = 0
total = 0
running_loss = 0.0
model.eval()
with torch.no_grad():
    for image, label in d_test:
        image, label = image.to(device), label.to(device)
        output = model(image)

        loss = loss_function(output.flatten(), label.float())
        running_loss += loss.item()

        output = (output > 0.5).flatten().int()
        correct += (output == label).sum().item()
        total += output.size(0)

print(f"Test Loss: {running_loss / len(d_test)}, Test Accuracy: {correct / total}")
print("-------------------------------------------------\n")
print("-------------------------------------------------")

print("Job ended at:", time.ctime())
