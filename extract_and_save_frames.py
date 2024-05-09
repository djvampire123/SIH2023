import os
import shutil
import cv2
import threading

def save_frame(frame, frame_number, extract_dir):
    try:
        frame_file = os.path.join(extract_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(frame_file, frame)
    except Exception as e:
        print(f"Error saving frame {frame_number}: {e}")

def extract_frames_and_copy_first_frame(video_path, extract_dir, copy_dir, frame_filename):
    # Ensure directories are set up
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    if not os.path.exists(copy_dir):
        os.makedirs(copy_dir)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    threads = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame in a separate thread
        thread = threading.Thread(target=save_frame, args=(frame, frame_count, extract_dir))
        thread.start()
        threads.append(thread)

        # Copy the first frame directly
        if frame_count == 0:
            dest_path = os.path.join(copy_dir, frame_filename)
            cv2.imwrite(dest_path, frame)

        frame_count += 1

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    cap.release()
    print(f"Processed {frame_count} frames.")

# Example usage
video_path = 'uploads/originalvideo.mp4'  # Replace with your video path
extract_dir = 'extracted_frames'          # Replace with your desired extract directory
copy_dir = 'static'                       # Replace with your desired copy directory
frame_filename = 'frame_0.jpg'            # Frame to copy

extract_frames_and_copy_first_frame(video_path, extract_dir, copy_dir, frame_filename)