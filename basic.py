import torch
import torch.nn as nn
import torch.optim as optim
import time

print("-------------------------------------------------")
print("Job started at:", time.ctime())
print("-------------------------------------------------\n")

print("--------------PyTorch Version Check--------------")
print("PyTorch Version:", torch.__version__)
print("-------------------------------------------------")


# Define the neural network using nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=10, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=1),
    nn.Sigmoid()
)

print("Defined Neural Network Architecture:")
print(model)

# Generate random input and target data
batch_size = 64
input_features = 10
inputs = torch.randn(batch_size, input_features)
targets = torch.randint(0, 2, (batch_size, 1)).float()

print("\nGenerated Random Inputs and Targets")

# Define loss function and optimizer
loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with learning rate of 0.01

print("Defined Loss Function and Optimizer")

# Training loop
epochs = 5
for epoch in range(epochs):
    # Forward pass: Compute predicted output by passing inputs to the model
    outputs = model(inputs)
    
    # Compute loss
    loss = loss_function(outputs, targets)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Demonstrate inference
with torch.no_grad():
    test_input = torch.randn(1, input_features)
    test_output = model(test_input)
    print(f"\nExample Inference Input: {test_input}")
    print(f"Example Inference Output (Probability of Class 1): {test_output.item()}")

