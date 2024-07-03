import cv2
import time
from video_input_module import capture_video

def adjust_dimensions_to_model_compatibility(width, height):
    """
    Adjust dimensions to be divisible by 32 for model compatibility.
    :param width: Original width of the frame.
    :param height: Original height of the frame.
    :return: Adjusted width and height.
    """
    adjusted_width = 32 * (width // 32)
    adjusted_height = 32 * (height // 32)
    return adjusted_width, adjusted_height

def preprocess_frame(frame, target_width, target_height, blur_ksize=(5, 5)):
    """
    Preprocess the frame for YOLO object detection.
    :param frame: Original frame from video feed.
    :param target_width: Desired width before adjusting for model compatibility.
    :param target_height: Desired height before adjusting for model compatibility.
    :param blur_ksize: Kernel size for Gaussian Blur.
    :return: Preprocessed frame and new resolution.
    """
    try:
        # Adjust target dimensions for model compatibility
        target_width, target_height = adjust_dimensions_to_model_compatibility(target_width, target_height)
        
        # Resize the frame to the adjusted dimensions
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Apply Gaussian Blur for noise reduction
        frame = cv2.GaussianBlur(frame, blur_ksize, 0)

        return frame, target_width, target_height
    except Exception as e:
        print(f"An error occurred during frame preprocessing: {e}")
        return None, None, None

def process_video(video_path, frame_handler, desired_fps, target_width, target_height):
    try:
        frame_interval = 1.0 / desired_fps
        details_printed = False

        for frame in capture_video(video_path):
            start_time = time.time()

            processed_frame, new_width, new_height = preprocess_frame(frame, target_width, target_height)
            if processed_frame is None:
                continue  # Skip frame processing if preprocessing failed

            if not details_printed:
                print(f"---New Feed Details---\nResolution: {new_width}x{new_height}, Frame Rate: {desired_fps} FPS")
                details_printed = True

            # frame_handler function is called to process each frame
            try:
                frame_handler(processed_frame, new_width, new_height)
            except Exception as e:
                print(f"An error occurred during frame handling: {e}")

            time.sleep(max(0, frame_interval - (time.time() - start_time)))
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
    finally:
        cv2.destroyAllWindows()
