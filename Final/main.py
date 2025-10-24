# main.py

import cv2
import argparse
import utils # Our custom utility functions
import matplotlib.pyplot as plt
import os

def plot_error_graphs(cte_data, he_data):
    """
    Displays plots of Cross-Track Error and Heading Error over time.
    """
    # Filter out 'None' values that may have occurred on frames
    # where detection or line fitting failed.
    clean_cte = [val for val in cte_data if val is not None]
    clean_he = [val for val in he_data if val is not None]
    
    frame_numbers_cte = range(len(clean_cte))
    frame_numbers_he = range(len(clean_he))

    plt.figure(figsize=(14, 7))

    # Plot 1: Cross-Track Error
    plt.subplot(1, 2, 1)
    plt.plot(frame_numbers_cte, clean_cte, label='Cross-Track Error', color='b')
    plt.xlabel('Frame Number')
    plt.ylabel('Error (pixels)')
    plt.title('Cross-Track Error over Time')
    plt.legend()
    plt.grid(True)

    # Plot 2: Heading Error
    plt.subplot(1, 2, 2)
    plt.plot(frame_numbers_he, clean_he, label='Heading Error', color='r')
    plt.xlabel('Frame Number')
    plt.ylabel('Error (degrees)')
    plt.title('Heading Error over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    print("Displaying error plots. Close the plot window to exit.")
    plt.show()

def process_video(video_path, output_clean_path=None, output_debug_path=None,beta=0.2,save_servo_frames_path=None, save_debug_frames_path=None):
    """
    Processes a video to detect crop row centerlines and saves two versions of the output:
    1. Clean (Servo) View: Robot vs. Path vectors and error values.
    2. Debug View: All detection visuals plus the fitted line.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # --- Setup Video Writers ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    clean_writer = None
    if output_clean_path:
        # The "clean" output will now be the new Servo View
        clean_writer = cv2.VideoWriter(output_clean_path, fourcc, fps, (frame_width, frame_height))
        print(f"Servo (clean) output will be saved to {output_clean_path}")

    debug_writer = None
    if output_debug_path:
        # The "debug" output will be the new detailed Debug Frame
        debug_writer = cv2.VideoWriter(output_debug_path, fourcc, fps, (frame_width, frame_height))
        print(f"Debug output will be saved to {output_debug_path}")
        
    if save_servo_frames_path:
        os.makedirs(save_servo_frames_path, exist_ok=True)
        print(f"Servo frames will be saved to {save_servo_frames_path}")
        
    if save_debug_frames_path:
        os.makedirs(save_debug_frames_path, exist_ok=True)
        print(f"Debug frames will be saved to {save_debug_frames_path}")


    prev_m = None 
    prev_b = None
    cte_holder = []
    he_holder = []
    # --- Process Video Frame by Frame ---
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break
        
        # --- 1. Main Processing Pipeline ---
        mask = utils.create_denoised_mask(frame)
        center_pts, left_pts, right_pts = utils.find_lane_and_centerline_points(mask)
        
        # --- 2. Fit Line and Calculate Errors ---
        cte, he, m, b = utils.calculate_errors_and_fit(center_pts, frame_width, frame_height,prev_m,prev_b,beta=beta)
        prev_m = m
        prev_b = b
        cte_holder.append(cte)
        he_holder.append(he)
        # --- 3. Generate both frames ---
        
        # Frame 1: Full debug view with points, strips, and fitted line
        debug_frame = utils.draw_debug_frame(frame, center_pts, left_pts, right_pts, m, b)
        
        # Frame 2: Minimal servo view with vectors and errors on black background
        servo_frame = utils.draw_servo_frame(frame, cte, he, m, b)

        # --- 4. Display the frames ---
        cv2.imshow('Debug Frame', debug_frame)
        cv2.imshow('Servo View', servo_frame)
        
        # --- 5. Write frames to output files if specified ---
        if clean_writer:
            clean_writer.write(servo_frame)
        if debug_writer:
            debug_writer.write(debug_frame)

        # if save_servo_frames_path:
        #     servo_filename = os.path.join(save_servo_frames_path, f"servo_frame_{frame_count:06d}.jpg")
        #     cv2.imwrite(servo_filename, servo_frame)
            
        if save_debug_frames_path and he is not None and he > 10:
            debug_filename = os.path.join(save_debug_frames_path, f"debug_frame_{frame_count:06d}.jpg")
            cv2.imwrite(debug_filename, debug_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # --- NEW: Increment frame counter ---
        frame_count += 1

    # --- Cleanup ---
    cap.release()
    if clean_writer:
        clean_writer.release()
    if debug_writer:
        debug_writer.release()
    cv2.destroyAllWindows()
    
    return cte_holder, he_holder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect centerline in agricultural video footage.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument(
        '-oc', '--output_clean', 
        type=str, 
        help="Path to save the clean servo output video."
    )
    parser.add_argument(
        '-od', '--output_debug', 
        type=str, 
        help="Path to save the detailed debug output video."
    )
    
    parser.add_argument(
        '--save_servo_frames', 
        type=str, 
        help="Path to a directory to save individual servo frames."
    )
    parser.add_argument(
        '--save_debug_frames', 
        type=str, 
        help="Path to a directory to save individual debug frames."
    )
    
    args = parser.parse_args()
    for i in [0.2]:
        cte_data, he_data = process_video(args.input, args.output_clean, args.output_debug,beta=i,save_servo_frames_path=args.save_servo_frames,save_debug_frames_path=args.save_debug_frames)
        plot_error_graphs(cte_data[100:],he_data[100:])
    
