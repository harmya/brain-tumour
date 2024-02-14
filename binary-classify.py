import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import torchvision
import numpy as np


# ----------------- Important Variables -----------------
batch_size = 16
learning_rate = 0.001
epochs = 10
# ------------------------------------------------------

#----------------- Loading Data ------------------------
training = torch.load('torch-dataset/training.pt')
validation = torch.load('torch-dataset/validation.pt')
testing = torch.load('torch-dataset/testing.pt')

d_train = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
d_vali = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)
d_test = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=False)
print("----------------- Data Loaded ---------------------")
print("Number of training samples: ")
print(len(d_train.dataset))
print("Number of validation samples: ")
print(len(d_vali.dataset))
print("Number of testing samples: ")
print(len(d_test.dataset))
print("--------------------------------------------------")
# ------------------------------------------------------

#------------------------ Model ----------------------------



