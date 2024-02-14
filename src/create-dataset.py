import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import torchvision
import numpy as np
import random
import os
import PIL.Image as Image 

# ----------------- Important Variables -----------------
training_yes_tumour_path = 'data-tumor/training/meningioma'
training_no_tumour_path = 'data-tumor/training/notumor'
testing_yes_tumour_path = 'data-tumor/testing/meningioma'
testing_no_tumour_path = 'data-tumor/testing/notumor'
# ------------------------------------------------------


#----------------- Getting Files -----------------------

training_yes_tumour_files = []
training_no_tumour_files = []
testing_yes_tumour_files = []
testing_no_tumour_files = []

for filename in os.listdir(training_yes_tumour_path):
    if filename.endswith(".jpg"):
        training_yes_tumour_files.append(filename)
for filename in os.listdir(training_no_tumour_path):
    if filename.endswith(".jpg"):
        training_no_tumour_files.append(filename)
for filename in os.listdir(testing_yes_tumour_path):
    if filename.endswith(".jpg"):
        testing_yes_tumour_files.append(filename)
for filename in os.listdir(testing_no_tumour_path):
    if filename.endswith(".jpg"):
        testing_no_tumour_files.append(filename)

print("Number of training yes tumour files: ")
print(len(training_yes_tumour_files))
print("Number of training no tumour files: ")
print(len(training_no_tumour_files))
print("Number of testing yes tumour files: ")
print(len(testing_yes_tumour_files))
print("Number of testing no tumour files: ")
print(len(testing_no_tumour_files))

# ------------------------------------------------------

#----------------- Creating Dataset ---------------------
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Grayscale()
])

training = []
testing = []

for filename in training_yes_tumour_files:
    img = Image.open(training_yes_tumour_path + '/' + filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    training.append([preprocess(img), 1])
for filename in training_no_tumour_files:
    img = Image.open(training_no_tumour_path + '/' + filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    training.append([preprocess(img), 0])

for filename in testing_yes_tumour_files:
    img = Image.open(testing_yes_tumour_path + '/' + filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    testing.append([preprocess(img), 1])
for filename in testing_no_tumour_files:
    img = Image.open(testing_no_tumour_path + '/' + filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    testing.append([preprocess(img), 0])

random.shuffle(training)
random.shuffle(testing)
split = int(0.95 * len(training))
validation = training[split:]
training = training[:split]

print("Number of training images: ")
print(len(training))
print("Number of validation images: ")
print(len(validation))
print("Number of testing images: ")
print(len(testing))

# ------------------------------------------------------

# ----------------- Save Dataset -------------------------
torch.save(training, 'torch-dataset/training.pt')
torch.save(validation, 'torch-dataset/validation.pt')
torch.save(testing, 'torch-dataset/testing.pt')
# ------------------------------------------------------

