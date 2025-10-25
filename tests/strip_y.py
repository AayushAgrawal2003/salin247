import cv2
import numpy as np
from scipy.signal import find_peaks

# --- Use the specified file paths ---
try:
    final_mask = cv2.imread('denoised_mask.jpg', cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread('/home/airlab/Desktop/aayush/Data/Images/Reference.png')
    
    if final_mask is None:
        print("Error: 'denoised_mask.jpg' not found or could not be read.")
        exit()
    if original_image is None:
        print("Error: '/home/airlab/Desktop/aayush/Data/Images/Reference.png' not found or could not be read.")
        exit()
        
    output_image = original_image.copy()
except Exception as e:
    print(f"An error occurred while loading images: {e}")
    exit()


# --- Define Parameters ---
height, width = final_mask.shape
strip_height = 80
y_horizon = int(height * 0.35)

centerline_points = []

# --- Iterate Through Strips from Bottom to Top ---
for y_end in range(height, y_horizon, -strip_height):
    y_start = max(y_horizon, y_end - strip_height)
    
    # Visualize the strip boundary
    cv2.line(output_image, (0, y_start), (width, y_start), (255, 255, 0), 1) # Cyan line

    strip_mask = final_mask[y_start:y_end, :]
    
    if np.sum(strip_mask) < 1000:
        continue

    vertical_projection = np.sum(strip_mask, axis=0)
    peaks, _ = find_peaks(vertical_projection, height=np.max(vertical_projection)*0.3, distance=40)

    if len(peaks) < 2:
        continue
    
    # --- Isolate and Mark Only the Left and Right Center Lanes ---
    image_center_x = width / 2
    left_peaks = peaks[peaks < image_center_x]
    right_peaks = peaks[peaks > image_center_x]

    if not left_peaks.any() or not right_peaks.any():
        continue
        
    # Get the specific x-coordinates for the two centermost rows
    center_left_row_x = left_peaks.max()
    center_right_row_x = right_peaks.min()
    
    y_center = (y_start + y_end) // 2
    
    # Mark the left and right lanes in a different color (Blue)
    cv2.circle(output_image, (center_left_row_x, y_center), 7, (255, 0, 0), -1) # Blue circle
    cv2.circle(output_image, (center_right_row_x, y_center), 7, (255, 0, 0), -1) # Blue circle

    # Calculate the midpoint for the final centerline
    x_center = (center_left_row_x + center_right_row_x) / 2
    centerline_points.append((int(x_center), y_center))


# --- Connect the Midpoints to Form the Final Centerline ---
if len(centerline_points) > 1:
    centerline_points.sort(key=lambda p: p[1])
    pts = np.array(centerline_points, np.int32)
    
    # Draw the final centerline in red
    cv2.polylines(output_image, [pts], isClosed=False, color=(0, 0, 255), thickness=3)

    cv2.imwrite('result_with_center_lanes_marked.jpg', output_image)
    print("Successfully generated image with center lanes marked.")
else:
    print("Could not generate enough points to create a centerline.")