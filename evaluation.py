# Load the trained neural network model
model = neural_network_v2.SecondCNN()
model.load_state_dict(torch.load(r'Depixelated_images\final_model.pth'))

# Prepare data for predictions
dataloader_validation = torch.utils.data.DataLoader(Dataloader_.test_set, shuffle=False, batch_size=1)
input_data = dataloader_validation
predicted = []

# Perform predictions using the loaded model
with torch.no_grad():
    for input1, input2, target in input_data:
        input1 = input1.double()
        input2 = input2.double()
        concatenated_image = torch.cat((input1, input2), dim=1)
        output = model(concatenated_image)
        predicted.append(output)

# Prepare target images and known arrays
targeted = []
known_array = []

with torch.no_grad():
    for input1, input2, target in input_data:
        input1 = input1.double()
        input2 = input2.double()
        target = target.double()
        targeted.append(target * 255)
        known_array.append(input2)

# Convert the targeted images and known arrays to appropriate formats
targeted_ = []
known_array_ = []

for i in targeted:
    i = torch.squeeze(i, dim=0)
    i = i.numpy().astype(np.uint8)
    targeted_.append(i)

for i in known_array:
    i = torch.squeeze(i, dim=0)
    i = i.numpy().astype(np.uint8)
    i = i.astype(bool)
    known_array_.append(i)

# Display the images and perform RMSE calculation
for pred, image, known in zip(predicted_, targeted_, known_array_):
    fig, axes = plt.subplots(ncols=3)
    axes[0].imshow(pred[0], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Predicted")
    axes[1].imshow(image[0], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Original")
    axes[2].imshow(known[0], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Known array")
    fig.tight_layout()
    plt.show()

# Define a function to calculate the Root Mean Squared Error (RMSE)
def calculate_rmse(array1, array2):
    squared_diff = (array1 - array2) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    return rmse

# Calculate RMSE for each image and print the results
rmse_lst = []
for x, y, z in zip(predicted_, targeted_, known_array_):
    x = x[~z]
    y = y[~z]
    rmse_ = calculate_rmse(x, y)
    rmse_lst.append(rmse_)
print(rmse_lst)