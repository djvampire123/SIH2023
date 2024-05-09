import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split()
        return np.array([int(coord) for coord in data])

# Set up paths
input_folder = 'uploads'  # Update this with the path to your folder
output_folder = 'uploads'  # Update this with the desired output path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the image
image = cv2.imread('extracted_frames/frame_0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Add the path to the SAM model
sys.path.append("..")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# Load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Set the image in the predictor
predictor.set_image(image)

# Process each file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith('coords') and file_name.endswith('.txt'):
        file_path = os.path.join(input_folder, file_name)

        # Read coordinates from the file
        coordinates = read_coordinates(file_path)

        # Create input box array
        input_box = np.array([coordinates[0], coordinates[1], coordinates[2], coordinates[3]])

        # Make predictions using the neural network model
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Save the mask to a file with the corresponding index
        output_file_path = os.path.join(output_folder, f'mask{file_name.split("coords")[1][0]}.npy')
        np.save(output_file_path, masks)

print("Processing complete.")
