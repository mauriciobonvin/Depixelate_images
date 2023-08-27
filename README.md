# Author

Mauricio Bonvin
27/8/23

# Machine Learning (ML) project: Depixelate_images

This project is ML project and contains a simple Convolutional Neural Network(CNN) that is trained to depixelate an input pixelated image.
The input images consisted in 34.635 pixelated images in grayscale.
The output images (test set) are depixelated images in greyscale.
The Root Mean Squared Error (RMSE) is calculated as a measure of similarity between the predicted and target images. 
Visualization of the predictions, original images, and pixelated area is also provided.

## Usage

1. Data Preparation: 
Prepare your training and validation datasets. The datasets should include pairs of input images, target images, and corresponding known arrays.

2. Training: 
Utilize the training loop to train the SecondCNN model. 
The training process involves specifying the number of epochs, providing the datasets, and saving the model's weights.

3. Inference and Evaluation: 
Load the trained model using the saved .pth file.
 Perform predictions on the validation dataset and calculate the RMSE between the predicted images and target images. 
Visualize the results using matplotlib.

## Dependencies

-Python 3.10

-PyTorch

-NumPy

-Matplotlib

## Structure
```
Image_depixelation
|- architecture.py
|    This code defines a neural network architecture using PyTorch's nn.Module as the base class.
|    The network consists of convolutional and deconvolutional (transpose convolution) layers,
|    a fully connected output layer, and activation functions. It takes an input with 2 channels (specified by n_in_channels)
|    and produces an output with a single channel, reshaped to match the input image dimensions.
|    The network is intended for image-based tasks like image transformation or generation.
|- data_preparation.py
|    This code defines a PyTorch Dataset class for working with image data.
|    It includes functionality for loading, transforming, pixelating, and preparing images.
|    The dataset is intended for tasks that involve pixelated image generation and completion.
|- main.py
|    This code defines a training loop for a neural network using PyTorch.
|    It trains the network on a training dataset, evaluates it on a validation dataset,
|    and tracks the training and validation losses over epochs. The script also contains a function to plot these losses.
|    The training loop includes early stopping to prevent overfitting.
|    The if __name__ == "__main__": block ensures that the code is only executed when the script is run as the main program.
|- evaluation.py
|    This code segment loads a trained neural network model, uses it to perform predictions on a validation dataset,
|    prepares target images and known arrays, displays images, calculates the Root Mean Squared Error (RMSE) for each image,
|    and prints the RMSE values. It also includes a function to calculate RMSE.
|- final_model.pth
|   The file stores the architecture of the neural network, including the layers, their configurations,
|   and any other components that define the model's structure.
|   The learned parameters of the model (weights and biases) are saved in the file.
|   These parameters are what the model has learned during training to make accurate predictions.
