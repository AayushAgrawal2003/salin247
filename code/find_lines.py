import cv2
import numpy as np
from scipy.signal import find_peaks

# --- Load previous results ---
final_mask = cv2.imread('denoised_mask.jpg', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('/home/airlab/Desktop/aayush/Data/Images/Reference.png')
output_image = original_image.copy() # Create a copy to draw on

# --- 1. Find Approximate Row Locations (Vertical Projection) ---
vertical_projection = np.sum(final_mask, axis=0)
peaks, _ = find_peaks(vertical_projection, height=np.max(vertical_projection)*0.4, distance=50)

# Identify the two center rows from the peaks
height, width = final_mask.shape
image_center_x = width / 2

left_peaks = peaks[peaks < image_center_x]
center_left_row_x = left_peaks.max() if len(left_peaks) > 0 else None

right_peaks = peaks[peaks > image_center_x]
center_right_row_x = right_peaks.min() if len(right_peaks) > 0 else None

if center_left_row_x is None or center_right_row_x is None:
    print("Error: Could not find two distinct center rows. Exiting.")
else:
    # --- 2. Create Masks for the Two Middle Rows (ROIs) ---
    roi_width = 150  # The width of the vertical strip to isolate each row. Tune if needed.
    
    # Create mask for the left center row
    left_roi_mask = np.zeros_like(final_mask)
    left_roi_start_x = max(0, int(center_left_row_x - roi_width / 2))
    left_roi_end_x = min(width, int(center_left_row_x + roi_width / 2))
    left_roi_mask[:, left_roi_start_x:left_roi_end_x] = 255
    left_lane_mask = cv2.bitwise_and(final_mask, left_roi_mask)

    # Create mask for the right center row
    right_roi_mask = np.zeros_like(final_mask)
    right_roi_start_x = max(0, int(center_right_row_x - roi_width / 2))
    right_roi_end_x = min(width, int(center_right_row_x + roi_width / 2))
    right_roi_mask[:, right_roi_start_x:right_roi_end_x] = 255
    right_lane_mask = cv2.bitwise_and(final_mask, right_roi_mask)

    # --- 3. Extract Points and Fit Curves ---
    # Get coordinates of white pixels for each isolated lane
    left_y_coords, left_x_coords = np.where(left_lane_mask > 0)
    right_y_coords, right_x_coords = np.where(right_lane_mask > 0)

    if left_y_coords.size > 10 and right_y_coords.size > 10: # Ensure enough points to fit
        # Fit a 2nd degree polynomial (a curve) to each set of points
        # We fit y vs x, so the polynomial can model the x-coordinate for any given y
        poly_left = np.polyfit(left_y_coords, left_x_coords, 2)
        poly_right = np.polyfit(right_y_coords, right_x_coords, 2)

        # --- 4. Calculate and Draw the Centerline ---
        centerline_points = []
        y_start_point = int(height * 0.40) # Start drawing from the horizon

        for y in range(y_start_point, height):
            # Calculate the x-position on each curve for the current y
            x_left = np.polyval(poly_left, y)
            x_right = np.polyval(poly_right, y)
            
            # Calculate the center point
            x_center = (x_left + x_right) / 2
            
            # IMPORTANT: Append points in (x, y) format for OpenCV drawing
            centerline_points.append((int(x_center), y))

        # Draw the final centerline as a series of connected lines (a polyline)
        if centerline_points:
            pts = np.array(centerline_points, np.int32)
            cv2.polylines(output_image, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
            
            # Optional: Draw the fitted lines for the individual rows for visualization
            left_lane_pts = np.array([(int(np.polyval(poly_left, y)), y) for y in range(y_start_point, height)], np.int32)
            right_lane_pts = np.array([(int(np.polyval(poly_right, y)), y) for y in range(y_start_point, height)], np.int32)
            cv2.polylines(output_image, [left_lane_pts], isClosed=False, color=(255, 0, 0), thickness=2) # Blue
            cv2.polylines(output_image, [right_lane_pts], isClosed=False, color=(255, 0, 0), thickness=2) # Blue

            cv2.imwrite('result_with_roi_fitted_centerline.jpg', output_image)
            print("Successfully generated centerline using the ROI method.")
    else:
        print("Not enough points found in the ROIs to fit curves.")
        
cv2.imshow("right",right_lane_mask)
cv2.imshow("left",left_lane_mask)
cv2.waitKey(0)