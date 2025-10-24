# utils.py

import cv2
import numpy as np
from scipy.signal import find_peaks

frame_crop = 0.4

def create_denoised_mask(frame):
    """
    Takes a video frame (BGR) and creates a binary mask of the green plants,
    ignoring the top portion of the image and removing noise.

    Args:
        frame (np.ndarray): The input image/frame from the video.

    Returns:
        np.ndarray: A denoised binary mask highlighting the plant rows.
    """
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])
    plant_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    height, width, _ = frame.shape
    top_mask = np.zeros((height, width), dtype=np.uint8)
    mask_start_row = int(height * frame_crop)
    top_mask[mask_start_row:, :] = 255

    final_mask = cv2.bitwise_and(plant_mask, top_mask)

    kernel = np.ones((5, 5), np.uint8)
    denoised_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    denoised_mask = cv2.morphologyEx(denoised_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return denoised_mask


def find_lane_and_centerline_points(mask):
    """
    Analyzes horizontal strips of a mask to find the center path and the lanes it's derived from.

    Args:
        mask (np.ndarray): The binary input mask from create_denoised_mask.

    Returns:
        tuple: A tuple containing (centerline_points, left_lane_points, right_lane_points).
    """
    height, width = mask.shape
    strip_height = 100
    y_horizon = int(height * frame_crop)

    centerline_points = []
    left_lane_points = []
    right_lane_points = []

    for y_end in range(height, y_horizon, -strip_height):
        y_start = max(y_horizon, y_end - strip_height)
        strip_mask = mask[y_start:y_end, :]

        if np.sum(strip_mask) < 1000:
            continue

        vertical_projection = np.sum(strip_mask, axis=0)
        peaks, _ = find_peaks(
            vertical_projection, 
            height=np.max(vertical_projection) * 0.3, 
            distance=40
        )

        if len(peaks) < 2:
            continue
        
        image_center_x = width / 2
        left_peaks = peaks[peaks < image_center_x]
        right_peaks = peaks[peaks > image_center_x]

        if not left_peaks.any() or not right_peaks.any():
            continue
            
        center_left_row_x = left_peaks.max()
        center_right_row_x = right_peaks.min()
    
        y_center = (y_start + y_end) // 2
        
        left_lane_points.append((center_left_row_x, y_center))
        right_lane_points.append((center_right_row_x, y_center))

        x_center = (center_left_row_x + center_right_row_x) / 2
        centerline_points.append((int(x_center), y_center))
    
    return centerline_points, left_lane_points, right_lane_points

def _normalize_angle(angle_deg):
    """Normalize an angle to the range [-180, 180]."""
    if angle_deg > 90:
        return angle_deg - 180
    else:
        return angle_deg
def draw_centerline(frame, centerline_points):
    """
    Draws ONLY the final centerline polyline onto a frame.
    """
    output_image = frame.copy()
    if len(centerline_points) > 1:
        centerline_points.sort(key=lambda p: p[1], reverse=True)
        pts = np.array(centerline_points, np.int32)
        cv2.polylines(output_image, [pts], isClosed=False, color=(0, 0, 255), thickness=4)
    return output_image


def _draw_original_debug_visuals(frame, centerline_points, left_points, right_points):
    """
    (Internal) Draws all the original visual aids: centerline, lane points, and strips.
    """
    # First, draw the basic polyline centerline
    output_image = frame.copy()
    
    # Draw the detected center points as red circles
    for point in centerline_points:
        cv2.circle(output_image, point, 5, (0, 0, 255), -1) # 5px radius red dots

    height, width, _ = frame.shape
    strip_height = 100
    y_horizon = int(height * frame_crop)

    # Draw the horizontal strip lines (cyan)
    for y_end in range(height, y_horizon, -strip_height):
        y_start = max(y_horizon, y_end - strip_height)
        cv2.line(output_image, (0, y_start), (width, y_start), (255, 255, 0), 1)

    # Draw the detected left and right lane points (blue circles)
    for point in left_points + right_points:
        cv2.circle(output_image, point, 7, (255, 0, 0), -1)

    # Draw robot's vertical center line (magenta)
    start_point = (width//2, 0)
    end_point = (width//2, height - 1)
    cv2.line(output_image, start_point, end_point, (255, 0, 255), 2)
    
    return output_image

def calculate_errors_and_fit(centerline_points, frame_width, frame_height,prev_m=None,prev_b=None,beta=0.2):
    """
    Fits a line to the centerline points and calculates CTE and HE.

    Args:
        centerline_points (list): List of (x, y) coordinates.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.

    Returns:
        tuple: (cte, he, m, b)
            cte (float): Cross-Track Error in pixels.
            he (float): Heading Error in degrees.
            m (float): Slope (dy/dx) of the fitted line.
            b (float): Y-intercept of the fitted line.
    """
    cte, he, m, b = None, None, None, None
    
    # We need at least 2 points to fit a line
    if len(centerline_points) < 2:
        return cte, he, m, b
    
    try:
        x_coords = np.array([p[0] for p in centerline_points])
        y_coords = np.array([p[1] for p in centerline_points])

        # Calculate Z-scores for the x-coordinates
        mean_x = np.mean(x_coords)
        std_x = np.std(x_coords)

        # Z-score calculation
        z_scores = (x_coords - mean_x) / std_x

        # Define threshold (typically 2 or 3 standard deviations)
        threshold = 1

        # Filter out points with z-score greater than the threshold
        filtered_points = [p for p, z in zip(centerline_points, z_scores) if abs(z) <= threshold]

        if len(filtered_points) < 4:
            filtered_points = [p for p, z in zip(centerline_points, z_scores) if abs(z) <= threshold+1]
            
            
            
        # Extract filtered x_coords and y_coords
        x_coords = np.array([p[0] for p in filtered_points])
        y_coords = np.array([p[1] for p in filtered_points])
        # Fit a line: y = m*x + b
        m, b = np.polyfit(x_coords, y_coords, 1)
        if prev_m is not None and abs(prev_m - m) > 10:
            m = prev_m
            b = prev_b
        
        if prev_m is not None and prev_b is not None:
            m = beta * m + (1-beta) * prev_m
            b = beta * b + (1-beta) * prev_b
        
        # print(m,b,len(filtered_points))
        # 1. Calculate Cross-Track Error (CTE)
        # Find the line's x-value at the bottom of the frame (y = frame_height)
        if m == 0:
            cte = None # Cannot calculate CTE for a horizontal line
        else:
            centerline_x_at_bottom = (frame_height - b) / m
            robot_x = frame_width / 2
            cte = robot_x - centerline_x_at_bottom

        # 2. Calculate Heading Error (HE)
        # The angle of the line relative to the horizontal (x-axis)
        path_angle_deg = np.degrees(np.arctan(m))
        
        # The robot's heading is straight up (negative y-direction),
        # which is -90 degrees from the horizontal x-axis.
        robot_heading_deg = -90.0
        
        # The heading error is the difference
        he = path_angle_deg - robot_heading_deg
        
        he = _normalize_angle(he)
        
        return cte, he, m, b

    except (np.linalg.LinAlgError, ZeroDivisionError) as e:
        print(f"Warning: Could not fit line. {e}")
        return None, None, None, None


def draw_debug_frame(frame, center_pts, left_pts, right_pts, m, b):
    """
    FRAME 1: Draws all debug info PLUS the fitted line.
    """
    # 1. Draw all the original detection visuals
    output_image = _draw_original_debug_visuals(frame, center_pts, left_pts, right_pts)
    
    # 2. Draw the fitted line/vector (if it exists)
    safe_m_threshold = 1e-5
    if m is not None and b is not None and np.isfinite(m) and np.isfinite(b) and abs(m) > safe_m_threshold:
        height, _, _ = frame.shape
        
        # Calculate start and end points of the fitted line
        y1 = height
        x1 = (y1 - b) / m
        
        y2 = int(height * frame_crop) # Horizon
        x2 = (y2 - b) / m
        # print((int(x1), int(y1)), (int(x2), int(y2)))
        # Draw the fitted line (yellow)
        # cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
        
        # Draw the fitted vector (green arrow)
        cv2.arrowedLine(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4, tipLength=0.1)
        
        start_point = (int(x1), 0)
        end_point = (int(x1), height - 1)
        cv2.line(output_image, start_point, end_point, (255, 0, 255), 2)
        

    return output_image


def draw_servo_frame(frame, cte, he, m, b):
    """
    FRAME 2: Draws minimal robot vs. path vectors and errors over the original video.
    """
    # 1. Use the original frame as the background
    height, width, _ = frame.shape
    output_image = frame.copy()

    # 2. Draw Robot Trajectory (Vertical magenta line)
    robot_x = width // 2
    cv2.line(output_image, (robot_x, height), (robot_x, 0), (255, 0, 255), 2)
    
    # 3. Draw Fitted Path Vector (Green arrow)
    safe_m_threshold = 1e-5
    if m is not None and b is not None and np.isfinite(m) and np.isfinite(b) and abs(m) > safe_m_threshold:
        y1 = height
        x1 = (y1 - b) / m
        
        y2 = int(height * 0.4) # Horizon
        x2 = (y2 - b) / m
        
        cv2.arrowedLine(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4, tipLength=0.1)

    # 4. Display Error Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cte_text = f"Cross-Track Error: {cte:.2f} px" if cte is not None else "CTE: N/A"
    he_text = f"Heading Error: {he:.2f} deg" if he is not None else "HE: N/A"
    
    # Add a semi-transparent background for the text
    cv2.rectangle(output_image, (40, 30), (600, 120), (0, 0, 0), -1)
    cv2.addWeighted(output_image, 1, output_image, 0.7, 0, output_image)
    
    cv2.putText(output_image, cte_text, (50, 50), font, 1, (255, 255, 0), 2)
    cv2.putText(output_image, he_text, (50, 100), font, 1, (255, 255, 0), 2)

    return output_image

