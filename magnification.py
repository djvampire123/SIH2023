import subprocess
import shutil
import os
import glob

def run_command():
    # Read alpha value from file
    with open('uploads/mag_fac.txt', 'r') as file:
        alpha_value = int(float(file.read().strip()))

    # Find all .npy mask files in the uploads directory
    mask_files = glob.glob('uploads/*.npy')
    mask_args = ["--mask_paths"] + mask_files

    command = [
        "python", "flowmag/inference.py",
        "--config", "flowmag/configs/alpha16.color10.yaml",
        "--frames_dir", "./extracted_frames",
        "--resume", "flowmag/checkpoints/raft_chkpt_00140.pth",
        "--save_name", "magnifiedvideo",
        "--alpha", str(alpha_value),
        "--soft_mask", "25",
        "--output_video"
    ] + mask_args

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Error in running command:")
        print(result.stderr)
        return
    else:
        print("Command executed successfully:")
        print(result.stdout)

    # Copy original video
    shutil.copy('uploads/originalvideo.mp4', 'static/originalvideo.mp4')

    # Convert processed video to MP4 MPEG-4
    processed_video_path = 'flowmag/inference/magnifiedvideo/overlayed_video.mp4'
    converted_video_path = 'static/processedvideo.mp4'
    
    convert_command = [
        "ffmpeg", "-i", processed_video_path,
        "-c:v", "libx264", "-crf", "10",
        "-preset", "slow", converted_video_path
    ]

    conversion_result = subprocess.run(convert_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if conversion_result.returncode != 0:
        print("Error in video conversion:")
        print(conversion_result.stderr)
    else:
        print("Video converted and copied successfully")

if __name__ == "__main__":
    run_command()