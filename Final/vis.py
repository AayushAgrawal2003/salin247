# visualize_journey.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import utils  # Your utility functions

def visualize_step_1_masking(frame, output_path):
    """Generates and saves the denoised mask."""
    mask = utils.create_denoised_mask(frame)
    cv2.imwrite(output_path, mask)
    print(f"Saved mask visualization to {output_path}")

def visualize_step_2_global_projection(mask, output_path):
    """Simulates the global projection method and saves the plot."""
    vertical_projection = np.sum(mask, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(vertical_projection)
    plt.title("Global Vertical Projection of Pixel Intensity")
    plt.xlabel("Image Width (pixels)")
    plt.ylabel("Sum of Pixel Intensity")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved global projection plot to {output_path}")

def visualize_step_3_strip_based_detection(frame, mask, output_path):
    """Visualizes the strips and detected local points."""
    center_pts, left_pts, right_pts = utils.find_lane_and_centerline_points(mask)
    
    # Use the internal drawing function from utils to show all visuals
    debug_image = utils._draw_original_debug_visuals(frame, center_pts, left_pts, right_pts)
    cv2.imwrite(output_path, debug_image)
    print(f"Saved strip-based detection visual to {output_path}")

def visualize_step_4_final_fit(frame, mask, output_path):
    """Shows the final fitted line after all filtering."""
    center_pts, left_pts, right_pts = utils.find_lane_and_centerline_points(mask)
    h, w, _ = frame.shape
    
    # Calculate errors to get the line parameters m, b
    _, _, m, b = utils.calculate_errors_and_fit(center_pts, w, h)
    
    # Use the final debug frame drawing function
    final_image = utils.draw_debug_frame(frame, center_pts, left_pts, right_pts, m, b)
    cv2.imwrite(output_path, final_image)
    print(f"Saved final line fit visual to {output_path}")

def visualize_step_5_temporal_smoothing(output_path):
    """Creates a plot to explain the concept of temporal smoothing."""
    # Generate some noisy data to simulate jittery heading error
    frames = np.arange(100)
    true_signal = 5 * np.sin(frames / 20)
    noise = np.random.normal(0, 1.5, 100)
    noisy_signal = true_signal + noise
    
    # Apply exponential moving average (the same logic as your code)
    beta = 0.2
    smoothed_signal = np.zeros_like(noisy_signal)
    smoothed_signal[0] = noisy_signal[0]
    for i in range(1, len(noisy_signal)):
        smoothed_signal[i] = beta * noisy_signal[i] + (1 - beta) * smoothed_signal[i-1]
        
    plt.figure(figsize=(12, 6))
    plt.plot(frames, noisy_signal, label='Noisy Measurement (Jittery)', alpha=0.6)
    plt.plot(frames, smoothed_signal, label='Smoothed Measurement (Beta=0.2)', linewidth=3, color='red')
    plt.title("Concept of Temporal Smoothing")
    plt.xlabel("Frame Number")
    plt.ylabel("Error Value (e.g., Heading Error)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved temporal smoothing plot to {output_path}")


if __name__ == '__main__':
    # --- Configuration ---
    REFERENCE_IMAGE_PATH = '../Images/Reference.png' # Make sure this file exists
    OUTPUT_DIR = 'output_visuals'
    
    # --- Setup ---
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"Error: Please create or place a '{REFERENCE_IMAGE_PATH}' in this directory.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        frame = cv2.imread(REFERENCE_IMAGE_PATH)
        mask = utils.create_denoised_mask(frame)

        # --- Generate Visuals for Each Step ---
        print("\n--- Generating Visualizations for Development Journey ---")
        visualize_step_1_masking(frame, os.path.join(OUTPUT_DIR, '1_denoised_mask.png'))
        visualize_step_2_global_projection(mask, os.path.join(OUTPUT_DIR, '2_global_projection_plot.png'))
        visualize_step_3_strip_based_detection(frame, mask, os.path.join(OUTPUT_DIR, '3_strip_based_points.png'))
        visualize_step_4_final_fit(frame, mask, os.path.join(OUTPUT_DIR, '4_final_fitted_line.png'))
        visualize_step_5_temporal_smoothing(os.path.join(OUTPUT_DIR, '5_temporal_smoothing_concept.png'))
        print("\n--- All visualizations saved in the 'output_visuals' directory! ---")