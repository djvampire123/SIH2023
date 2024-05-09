import cv2
import numpy as np

def read_amplification_factor(file_path):
    try:
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    except Exception as e:
        print(f"Error reading amplification factor from file: {e}")
        return 5  # Default value if file read fails

def process_video(video_path):
    amplification_factor = read_amplification_factor('uploads/amp_fac.txt')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter('output_heatmap24.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    heatmap_accumulator = np.zeros((frame_height, frame_width), np.float32)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if ret:
            frame_diff = cv2.absdiff(curr_frame, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            normalized_diff = cv2.normalize(gray_diff, None, 0, 255 * amplification_factor, cv2.NORM_MINMAX)

            heatmap = cv2.applyColorMap(normalized_diff, cv2.COLORMAP_JET)
            heatmap_accumulator += normalized_diff

            overlaid_frame = cv2.addWeighted(curr_frame, 0.7, heatmap, 0.3, 0)
            out.write(overlaid_frame)

            prev_frame = curr_frame
        else:
            break

    # Generate the final heatmap image
    average_heatmap = heatmap_accumulator / frame_count
    average_heatmap = np.clip(average_heatmap, 0, 255).astype(np.uint8)
    final_heatmap = cv2.applyColorMap(average_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite('static/final_heatmap.jpg', final_heatmap)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video('static/processedvideo.mp4')
