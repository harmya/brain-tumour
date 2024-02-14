import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print("------------Check if CUDA is available-------------")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

print("------------Important Variables----------------")
batch_size = 8
learning_rate = 0.01
epochs = 10
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
training = training[:100]
validation = torch.load('torch-dataset/validation.pt')
testing = torch.load('torch-dataset/testing.pt')

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
    nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1),

    nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1),

    nn.Flatten(),
    nn.Linear(64 * 254 * 254, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# model.to(device)
print(model)
print("-------------------------------------------------\n")


print("--------------Loss Function and Optimizer---------")
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print("Loss Function:", loss_function)
print("Optimizer:", optimizer)
print("-------------------------------------------------\n")



print("--------------Training Loop-----------------------")
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for image, label in d_train:
       # image, label = image.to(device), label.to(device)
        
        output = model(image)
        loss = loss_function(output, label.float().unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # check validation loss and accuracy
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        all_val_loss = []
        for image, label in d_vali:
 #           image, label = image.to(device), label.to(device)
            output = model(image)
            val_loss = loss_function(output, label.float().unsqueeze(1))
            print(output)
            print(label)
            predicted = torch.round(output).squeeze(1).int()
            print(predicted)
            total += predicted.size(0)
            print(predicted.size(0))
            correct += (predicted == label).sum().item()
            print((predicted == label).sum().item())
        print(f"Epoch {epoch+1}, validation loss: {val_loss.item()}, validation accuracy: {100 * correct / total}%")
print("-------------------------------------------------\n")


print("--------------Testing Loop------------------------")
correct = 0
total = 0
with torch.no_grad():
    for image, label in d_test:
        image, label = image.to(device), label.to(device)
        output = model(image)
        predicted = torch.round(output)
        total += 1
        correct += 1 if predicted == label else 0
print(f"Test Accuracy: {100 * correct / total}%")
print("-------------------------------------------------\n")
print("-------------------------------------------------")

print("Job ended at:", time.ctime())
