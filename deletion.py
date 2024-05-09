import os
import shutil
import glob

def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder not found: {folder_path}")

def delete_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File not found: {file_path}")

def delete_folders_starting_with(directory, prefix):
    for folder in glob.glob(os.path.join(directory, prefix + '*')):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")

# Delete 'uploads' folder and 'extracted_frames' folder
delete_folder('uploads')
delete_folder('extracted_frames')
delete_folder('processed')
delete_folder('flowmag/inference')

# Delete specific files in the 'static' directory
files_to_delete = [
    'static/final_heatmap.jpg',
    'static/frame_0.jpg',
    'static/originalvideo.mp4',
    'static/processedvideo.mp4'
]

for file_path in files_to_delete:
    delete_file(file_path)

# Delete all folders in 'static' starting with 'patch_'
delete_folders_starting_with('static', 'patch_')

print("Cleanup completed.")
