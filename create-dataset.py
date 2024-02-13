import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import PIL.Image as Image 

# ----------------- Important Variables -----------------
training_yes_tumour_path = 'data-tumor/training/meningioma'
training_no_tumour_path = 'data-tumor/training/notumor'
testing_yes_tumour_path = 'data-tumor/testing/meningioma'
testing_no_tumour_path = 'data-tumor/testing/notumor'

kernel_size = 5
padding_dimension = (kernel_size - 1) // 2
print("Padding dimension: ")
print(padding_dimension)
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
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = img.convert('L')
    img = torch.tensor(np.array(img))
    img = img / 255
    return img

training_yes_tumour_images = []
training_no_tumour_images = []

testing_yes_tumour_images = []
testing_no_tumour_images = []

for filename in training_yes_tumour_files:
    training_yes_tumour_images.append(process_image(training_yes_tumour_path + '/' + filename))
for filename in training_no_tumour_files:
    training_no_tumour_images.append(process_image(training_no_tumour_path + '/' + filename))

for filename in testing_yes_tumour_files:
    testing_yes_tumour_images.append(process_image(testing_yes_tumour_path + '/' + filename))
for filename in testing_no_tumour_files:
    testing_no_tumour_images.append(process_image(testing_no_tumour_path + '/' + filename))

training_yes_tumour_labels = torch.ones(len(training_yes_tumour_images))
training_no_tumour_labels = torch.zeros(len(training_no_tumour_images))
training_images = training_yes_tumour_images + training_no_tumour_images
training_labels = torch.cat((training_yes_tumour_labels, training_no_tumour_labels))
training_dataset = list(zip(training_images, training_labels))
random.shuffle(training_dataset)

testing_yes_tumour_labels = torch.ones(len(testing_yes_tumour_images))
testing_no_tumour_labels = torch.zeros(len(testing_no_tumour_images))
testing_images = testing_yes_tumour_images + testing_no_tumour_images
testing_labels = torch.cat((testing_yes_tumour_labels, testing_no_tumour_labels))
testing_dataset = list(zip(testing_images, testing_labels))
random.shuffle(testing_dataset)

print("Number of training images: ")
print(len(training_images))
print("Number of testing images: ")
print(len(testing_images))

# ------------------------------------------------------

# ----------------- Add Zero Padding ---------------------
for i in range(len(training_dataset)):
    training_dataset[i] = (F.pad(training_dataset[i][0], (padding_dimension, padding_dimension, padding_dimension, padding_dimension)), training_dataset[i][1])

for i in range(len(testing_dataset)):
    testing_dataset[i] = (F.pad(testing_dataset[i][0], (padding_dimension, padding_dimension, padding_dimension, padding_dimension)), testing_dataset[i][1])

print("Shape after padding: ")
print(training_dataset[0][0].shape)

# ------------------------------------------------------

# ----------------- Save Dataset -------------------------
torch.save(training_dataset, 'torch-dataset/training-dataset.pt')
torch.save(testing_dataset, 'torch-dataset/testing-dataset.pt')
# ------------------------------------------------------

