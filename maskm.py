import numpy as np

# Replace 'path/to/mask.npy' with the actual path to your 'mask.npy' file
mask_path = 'uploads/mask1.npy'

# Load the .npy file
mask = np.load(mask_path)

# Print the shape of the array
print("Shape of mask1.npy:", mask.shape)