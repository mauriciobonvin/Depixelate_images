# Mauricio Bonvin
# Matrikel-Nr 12146801
# python 3.10

import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings


# Define the training loop function
def training_loop(network: torch.nn.Module,
                  train_data: torch.utils.data.Dataset,
                  eval_data: torch.utils.data.Dataset,
                  num_epochs: int,
                  save_path: str,
                  device: str = "cuda",
                  show_progress: bool = False,
                  ) -> tuple[list, list]:

    # Set the device for training
    device = torch.device(device)

    # Check if CUDA is available, otherwise fall back to CPU
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Assign variables
    model = network
    model.to(device)
    train = train_data
    val = eval_data
    epochs = num_epochs

    # Create data loaders for training and validation data
    dataloader_train = torch.utils.data.DataLoader(train, shuffle=True, batch_size=32, num_workers=2)
    dataloader_validation = torch.utils.data.DataLoader(val, shuffle=False, batch_size=16)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the Mean Squared Error loss function
    loss_function = torch.nn.MSELoss()

    # Lists to store epoch losses for training and validation
    epoch_losses_train = []
    epoch_losses_val = []

    # Loop through epochs
    for epoch in tqdm(range(epochs), disable=not show_progress):
        mbl_train = [] # Mini-batch losses for training
        mbl_val = [] # Mini-batch losses for validation

        # Initialize counter and minimum loss for early stopping
        counter = 0
        min_loss = float()

        # Set the model to training mode
        model.train()

        # Training loop
        for input1, input2, target in tqdm(dataloader_train, leave=False, disable=not show_progress):
            input1 = input1.double()
            input2 = input2.double()
            target = target.double()

            # Concatenate the image tensors along the channel dimension
            concatenated_image = torch.cat((input1, input2), dim=1)
            concatenated_image = concatenated_image.to(device)
            target = target.to(device)

            # Forward pass, compute loss, and update the model
            output = model(concatenated_image)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mbl_train.append(loss.item())

        # Set the model to evaluation mode
        model.eval()

        # Validation loop
        with torch.no_grad():
            for input1e, input2e, targete in tqdm(dataloader_validation, leave=False, disable=not show_progress):
                input1e = input1e.double()
                input2e = input2e.double()
                targete = targete.double()

                # Concatenate the image tensors along the channel dimension
                concatenated_image = torch.cat((input1e, input2e), dim=1)
                concatenated_image = concatenated_image.to(device)
                targete = targete.to(device)

                # Forward pass and compute validation loss
                output_test = model(concatenated_image)
                loss_test = loss_function(output_test, targete)
                mbl_val.append(loss_test.item())

        # Calculate mean values for training and validation losses
        epoch_losses_train.append(np.mean(mbl_train))
        eval_loss_val = np.mean(mbl_val)
        epoch_losses_val.append(eval_loss_val)

        # Early stopping mechanism
        if eval_loss_val < min_loss:
            min_loss = eval_loss_val
            counter = 0
        else:
            counter += 1
        if counter >= 3:
            print("No new minimal evaluation loss was achieved in the last 3 epochs")
            break

        # Save model weights
        torch.save(model.state_dict(), save_path)

    return epoch_losses_train, epoch_losses_val

# Define a function to plot training and evaluation losses
def plot_losses(train_losses: list, eval_losses: list):
    steps = range(1, len(train_losses)+1)
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, eval_losses, label='Eval Loss')
    plt.title('Training and Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Import required modules
    import data_preparation
    import architecture
    torch.random.manual_seed(0)

    # Load training and validation data
    train_data_, val_data_ = data_preparation.training_set, data_preparation.validation_set

    # Create an instance of the neural network
    network_ = architecture.SecondCNN()

    # Run the training loop and obtain losses
    train_losses_, eval_losses_ = training_loop(network_, train_data_, val_data_, num_epochs=20, save_path=r"Depixelated_images\final_model.pth", show_progress=True)

    # Plot the training and evaluation losses
    plot_losses(train_losses_, eval_losses_)