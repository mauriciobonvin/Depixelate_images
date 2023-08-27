# Mauricio Bonvin
# Matrikel-Nr 12146801
# python 3.10

import torch
import torch.nn as nn


# Define a neural network architecture class
class SecondCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_kernels: int = 32, kernel_size: int = 3):
        super().__init__()

        # Define the dimensions of the input image
        image_height = 64
        image_width = 64
        output_size = image_height * image_width

        # Convolutional layers
        self.conv1 = nn.Conv2d(n_in_channels, n_kernels, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels, kernel_size=kernel_size, stride=1)
        self.conv3 = nn.Conv2d(n_kernels, n_kernels, kernel_size=kernel_size, stride=1)

        # Deconvolutional layers (transpose of convolutional layers)
        self.deconv1 = nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=kernel_size, stride=1)
        self.deconv2 = nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=kernel_size, stride=1)
        self.deconv3 = nn.ConvTranspose2d(n_kernels, 1, kernel_size=kernel_size, stride=1)

        # Fully connected layer for final output
        self.output_layer = nn.Linear(1 * image_height * image_width, output_size)

        # Activation function
        self.relu = nn.ReLU()

        # Set certain layers to use double precision
        self.output_layer.weight = nn.Parameter(self.output_layer.weight.double())
        self.output_layer.bias = nn.Parameter(self.output_layer.bias.double())
        self.conv1.double()
        self.conv2.double()
        self.conv3.double()
        self.deconv1.double()
        self.deconv2.double()
        self.deconv3.double()

    def forward(self, x):
        # Forward pass through the network

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.conv3(x))  # Apply ReLU activation after the last convolution

        # Apply deconvolutional layers (transposed convolution)
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        x = self.deconv3(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply the fully connected output layer
        x = self.output_layer(x)

        # Reshape the output to match the image dimensions
        x = x.view(x.size(0), 1, 64, 64)

        return x
