import cv2
import queue

def capture_video(video_feed, frame_queue_size=10, frame_skip=0):
    """
    Captures video frames from a specified video feed, managing frame rate dynamically by skipping frames.

    :param video_feed: The video source (e.g., a file path or RTSP URL).
    :param frame_queue_size: The maximum size of the frame queue.
    :param frame_skip: Number of frames to skip between each processed frame. A value of 0 means no skipping.
    """
    try:
        cap = cv2.VideoCapture(video_feed, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IOError("Error: Could not open video stream or file")
        
        print_original_feed_details(cap)
        
        frame_queue = queue.Queue(maxsize=frame_queue_size)
        frame_count = 0  # Initialize frame count
        is_rtsp = video_feed.startswith("rtsp://")
        if is_rtsp:
            frame_skip = 3  # Custom frame skip for RTSP to handle latency
        else:
            frame_skip = 0  # No frame skip for regular video files

        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Exit loop if no frame is returned

            frame_count += 1

            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                continue  # Skip frames as per frame_skip parameter

            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                yield frame_queue.get()
                frame_queue.put(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Allow early exit on 'q' key press

    except IOError as e:
        print(f"IOError encountered: {e}")
    except KeyboardInterrupt:
        print("Processing stopped by user - video_input_module.py")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        # Ensure all remaining frames in the queue are yielded
        while 'frame_queue' in locals() and not frame_queue.empty():
            yield frame_queue.get()

def print_original_feed_details(cap):
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"---Original Feed Details---\nResolution: {width}x{height}, Frame Rate: {fps} FPS")

        # Print total frames if available
        if total_frames > 0:
            print(f"Total Frames: {int(total_frames)}")
        else:
            print("Total Frames: Not available (live stream or unsupported format)")
    except Exception as e:
        print(f"Error retrieving video feed details: {e}")
