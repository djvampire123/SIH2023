import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.fft import fft


def resize_frame(frame, scale=1):
    """
    Resize the frame by a given scale.
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def calculate_displacement(centers):
    """Calculate displacement between consecutive centers."""
    displacements = [np.linalg.norm(centers[i] - centers[i - 1]) for i in range(1, len(centers))]
    return displacements



def main(bbox, save_dir, video_path):
    # Video capture
    if not os.path.exists(video_path):
        print(f"Video file not found at {video_path}")
        return

    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if not ret:
        print("Failed to read video")
        return
    
    # Specify the directory to save the plots
    '''save_dir = 'D:\\sih\\freq_plots_V2\\result'    # Ensure the directory exists, create if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)'''

    # Remove previous plot files in the directory
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)



    # Resize frame for ROI selection
    #resized_frame = resize_frame(frame, scale=0.5)
    #bbox = cv2.selectROI(resized_frame, False)

    # Scale back the bounding box coordinates to original frame size
    #scale_factor = frame.shape[1] / resized_frame.shape[1]
    #bbox = tuple([int(v * scale_factor) for v in bbox])

    
    '''
    cord_path = 'D:\\sih\\freq_plots_V2.9\\data\\cords.txt'
    with open(cord_path, 'r') as file:
        numbers_str = file.read()
    bbox = tuple(map(int, numbers_str.split(', ')))
    '''
    
    # Initialize tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    physical_width_cm = 2 # actual width in cm
    physical_height_cm = 2 # actual height in cm

# Calculate scale factors
    #scale_factor_width = physical_width_cm / bbox[2]
    #scale_factor_height = physical_height_cm / bbox[3]

    centers = []  # List to store center coordinates of the object
    frame_number = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1

        # Resize frame for display
        resized_frame = resize_frame(frame, scale=1)

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            # Calculate center of the bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append((frame_number, center_x, center_y))

            # Draw rectangle for visualization
            x1, y1, x2, y2 = [int(v) for v in bbox]
            #cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


            # Display the frame
            #cv2.imshow('Object Tracking', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Perform Frequency Analysis

    centers = np.array(centers)
    x_coords = centers[:, 1]
    y_coords = centers[:, 2]

    # Normalize coordinates
    x_coords -= np.mean(x_coords)
    y_coords -= np.mean(y_coords)

    # Get the frame rate from the video or set a default value
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print("Warning: Frame rate could not be determined. Using default value of 30 FPS.")
        frame_rate = 30  # Default frame rate





    #Displacement with time
    centers = np.array(centers)
    x_coords = centers[:, 1]
    y_coords = centers[:, 2]

    # Calculate displacements
    displacements = calculate_displacement(centers[:, 1:3])  # Using only x and y coordinates

    

    # Assuming `centers` stores the center coordinates in pixels for each frame
    # Initialize a list to store displacement in physical units
    displacements_cm = []

    old_center_x, old_center_y = None, None
    for frame_number, center_x, center_y in centers:
        if old_center_x is not None and old_center_y is not None:
            # Calculate displacement in pixels
            dx_px = center_x - old_center_x
            dy_px = center_y - old_center_y

            # Convert displacement to centimeters
            dx_px = dx_px
            dy_px = dy_px 

            # Store the displacement
            displacements_cm.append((frame_number, dx_px, dy_px))

        # Update the old center for the next iteration
        old_center_x, old_center_y = center_x, center_y


    # After initializing the VideoCapture object
    video = cv2.VideoCapture(video_path)

    # Get frame rate of the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)


    # Convert frame numbers to time in seconds
    times_in_seconds = [frame_number / frame_rate for frame_number, _, _ in displacements_cm]

    # Example: Plotting displacement over time in seconds
    dxs, dys = zip(*[(dx, dy) for _, dx, dy in displacements_cm])

    plt.figure(figsize=(10, 4))

    min_dx, max_dx = min(dxs), max(dxs)
    min_dy, max_dy = min(dys), max(dys)

    # Determine the wider range to set as the limit for both plots
    min_displacement = min(min_dx, min_dy)
    max_displacement = max(max_dx, max_dy)

    # Axis limits for both plots
    displacement_limits = (min_displacement, max_displacement)

    # Plotting
    plt.figure(figsize=(10, 4))

    # Plotting X displacement
    plt.subplot(1, 2, 1)
    plt.plot(times_in_seconds, dxs, label='X Displacement')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (px)')
    plt.title('Displacement in X Direction')
    plt.legend()
    plt.ylim(displacement_limits)

    # Plotting Y displacement
    plt.subplot(1, 2, 2)
    plt.plot(times_in_seconds, dys, label='Y Displacement')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (px)')
    plt.title('Displacement in Y Direction')
    plt.legend()
    plt.ylim(displacement_limits)

    plot_file = os.path.join(save_dir, 'two_dim.png')  # Change the file format if needed
    plt.savefig(plot_file)
    #plt.show()



    # Perform FFT
    displacement_fft = fft(dxs)  # or dys, depending on the motion direction

    # Get the power spectrum
    powers = np.abs(displacement_fft)

    # Frequency bins
    n = len(dxs)
    freq_bins = np.fft.fftfreq(n, d=1/frame_rate)

    # Find the frequency with the highest power
    dominant_frequency = freq_bins[np.argmax(powers)]
    print(f"Frequency: {dominant_frequency}Hz")

    
    freq_path = os.path.join(save_dir, 'freq.txt')

    freq = f"{dominant_frequency} Hz"
    with open(freq_path, 'w') as file:
        file.write(freq)
    
    

    # Convert centers to a numpy array for easier manipulation
    centers = np.array(centers)  # Replace with actual data
    frame_numbers = centers[:, 0] / frame_rate  # Convert frame numbers to time in seconds
    x_coords = centers[:, 1]
    y_coords = centers[:, 2]

    # Determine the wider range to set as the limit for both X and Y axes
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_coord = min(min_x, min_y)
    max_coord = max(max_x, max_y)

    # Create the 3D plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=frame_numbers,
        z=y_coords,
        mode='lines',
        line=dict(
            width=2,
            color='blue'
        )
    )])

    fig.update_layout(
        title='3D Object Tracking',
        scene=dict(
            xaxis=dict(range=[min_coord, max_coord], title='X Axis'),
            yaxis=dict(title='Time'),
            zaxis=dict(range=[min_coord, max_coord], title='Y Axis')
        )
    )

    # Save the plot
    plot_file = os.path.join(save_dir, 'three_dim.html')  # Change the file format if needed
    fig.write_html(plot_file)
    #fig.show()


    # Performing FFT on the simulated data
    displacement_fft = fft(dxs)
    powers = np.abs(displacement_fft)  # Power spectrum
    frame_rate = 30  # Assuming a frame rate of 30 fps for frequency calculation
    freq_bins = np.fft.fftfreq(n, d=1/frame_rate)

    # Plotting the FFT spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq_bins, powers, color='blue')
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True)
    plt.xlim(left=0)
    plot_file = os.path.join(save_dir, 'fft_spectrum.png')
    plt.savefig(plot_file)
    #plt.show()

    freq_bins = np.fft.fftfreq(n, d=1/frame_rate)

    # Plotting the frequency-amplitude graph
    plt.figure(figsize=(10, 6))
    plt.plot(freq_bins, displacement_fft.real, color='blue')
    plt.title("Frequency-Amplitude Graph")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(left=0)
    plot_file = os.path.join(save_dir, 'freq_amp.png')
    plt.savefig(plot_file)
    

    
if __name__ == "__main__":
    video_path = 'static/processedvideo.mp4'
    cord_path = 'uploads'
    patches_directory = 'static'
    os.makedirs(patches_directory, exist_ok=True)
i = 1  # Initialize i outside the loop
for filename in os.listdir(cord_path):
    if filename.startswith("coords") and filename.endswith(".txt"):
        file_path = os.path.join(cord_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                file_data = file.read()
                bbox = tuple(map(int, file_data.split(' ')))
                patch_number = i 
                patch_directory = os.path.join(patches_directory, f'patch_{patch_number}')
                os.makedirs(patch_directory, exist_ok=True)
                main(bbox, patch_directory, video_path)
            i += 1  # Increment i only for valid files