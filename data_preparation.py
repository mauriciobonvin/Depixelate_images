# Mauricio Bonvin
# Matrikel-Nr 12146801
# python 3.10

import random
from torch.utils.data import Dataset
from PIL import Image
import glob
import os.path
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.transforms import InterpolationMode
from torch.utils.data import Subset
import numpy as np
import torch


class ImageDataset(Dataset):

    def __init__(self, image_dir, image_size=64, train_split=0.7, val_split=0.1):

        # Initialize the dataset with given parameters
        self.image_dir = image_dir
        self.image_size = image_size

        # Get the absolute path of the input directory
        input_dir = os.path.abspath(self.image_dir)
        os.chdir(input_dir)

        # Verify if the input directory exists
        try:
            verify_path = os.path.exists(self.image_dir)
            if not verify_path:
                raise ValueError
        except ValueError as ex:
            print("the input directory does not exists!")
            raise ex

        # Collect image file paths ending with specific extensions
        file_list = glob.glob(os.path.join("**"), root_dir=input_dir,
                              recursive=True)  # requires python 3.10 for root_dir to work
        self.abs_path_file = []
        for i in file_list:
            if i.endswith((".jpg", ".JPG", ".jpeg", ".JPEG")):
                self.abs_path_file.append(os.path.abspath(i))

        # Create a sorted copy of the image file paths
        self.sorted_abs_path_file = self.abs_path_file.copy()
        self.sorted_abs_path_file.sort()  # sorted path list in ascending order

        # Split the dataset into train, validation, and test sets
        self.train_split = train_split
        self.val_split = val_split
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()

    def _split_dataset(self):

        # Calculate sizes for train, validation, and test sets
        dataset_size = len(self.sorted_abs_path_file)
        train_size = int(dataset_size * self.train_split)
        val_size = int(dataset_size * self.val_split)
        test_size = dataset_size - train_size - val_size

        # Create indices for train, validation, and test sets
        indices = list(range(dataset_size))
        train_indices, val_indices, test_indices = random_split(indices, [train_size, val_size, test_size])

        return train_indices, val_indices, test_indices

    def pixelate(self, image, x, y, width, height, size):
        # Apply pixelation to a specified region of the image
        image = image.copy()
        curr_x = x

        while curr_x < x + width:
            curr_y = y
            while curr_y < y + height:
                block = (
                    ..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
                image[block] = image[block].mean()
                curr_y += size
            curr_x += size

        return image

    def prepare_image(self, image, x, y, width, height, size):
        # Prepare the image by pixelating a specific region

        # Validate input values
        if image.ndim < 3 or image.shape[-3] != 1:
            raise ValueError(f"image must have shape (..., 1, H, W), {image.ndim}")
        if width < 2 or height < 2 or size < 2:
            raise ValueError("width/height/size must be >= 2")
        if x < 0 or (x + width) > image.shape[-1]:
            raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
        if y < 0 or (y + height) > image.shape[-2]:
            raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")

        # Define the area to pixelate
        area = (..., slice(y, y + height), slice(x, x + width))
        # Apply pixelation
        pixelated_image = self.pixelate(image, x, y, width, height, size)

        known_array = np.ones_like(image, dtype=bool)

        known_array[area] = False

        return pixelated_image, known_array

    def __getitem__(self, index_: int):
        # Get data for a specific index

        # Get the file path of the image
        file_path = self.sorted_abs_path_file[index_]

        # Open and transform the image
        image = Image.open(file_path)
        resize_transforms = transforms.Compose(
            [transforms.Resize(size=self.image_size, interpolation=InterpolationMode.BILINEAR),
             transforms.CenterCrop(size=(self.image_size, self.image_size)),
             transforms.Grayscale(),
             ])
        image = resize_transforms(image)

        # Normalize the image and add a brightness dimension
        image = np.array(image, dtype=np.uint8) / 255
        image = image[np.newaxis, :, :]

        # Generate random parameters for pixelation
        width = random.randint(4, 32)
        height = random.randint(4, 32)
        x = random.randint(0, 64 - width)
        y = random.randint(0, 64 - height)
        size = random.randint(4, 16)

        # Prepare the pixelated image and known array
        pixelated_image, known_array = self.prepare_image(image, x, y, width, height, size)

        return pixelated_image, known_array, image

    def __len__(self):
        # Return the length of the dataset
        return len(self.sorted_abs_path_file)


# instance the class (set path to a folder with images)
imgs = ImageDataset(
    "Depixelate_images\\training",
    64,
    0.7,
    0.1)

# Subset dataset into train, validation and test
training_set = Subset(imgs, indices=imgs.train_indices)
validation_set = Subset(imgs, indices=imgs.val_indices)
test_set = Subset(imgs, indices=imgs.test_indices)


'''
# visualize shapes and images
for image, pixelated_image, known_array in imgs:
    print(f"image:{image.shape},pixel:{pixelated_image.shape}, knwon: {known_array.shape}")

import matplotlib.pyplot as plt

for image, pixelated_image, known_array, target_array in imgs:
    fig, axes = plt.subplots(ncols=4)
    axes[0].imshow(image[0], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Pixelated_array")
    axes[2].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Known_array")
    axes[3].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("target_array")
    fig.tight_layout()
    plt.show()

'''
