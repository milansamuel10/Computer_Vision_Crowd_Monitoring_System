import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.solutions import heatmap
from preprocessing_module import preprocess_frame, process_video
from congestion_module import overlay_congestion_info, check_detection_threshold
from collections import defaultdict
import time
import datetime
import matplotlib.pyplot as plt
import traceback

class ObjectDetector:
    def __init__(self, model_path, detection_threshold, video_path, results_filename, target_width, \
                target_height, fps, confidence, generate_heatmap, heatmap_filename, annotated_filename,\
                summary_filename, total_objects_plot_filename, threshold_plot_filename, movement_plot_filename):
        
        self.model = YOLO(model_path)
        self.detection_threshold = detection_threshold
        self.video_path = video_path
        self.results_filename = results_filename
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.confidence = confidence
        self.generate_heatmap = generate_heatmap
        self.heatmap_filename = heatmap_filename
        self.annotated_filename = annotated_filename
        self.summary_filename = summary_filename
        self.total_objects_plot_filename = total_objects_plot_filename
        self.threshold_plot_filename = threshold_plot_filename 
        self.movement_plot_filename = movement_plot_filename
        self.device = None
        self.set_device()
        self.model.to(self.device)
        
    def set_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    def detect_objects_in_video(self, video_path):
        stop_processing = False
        total_frames_processed = 0
        total_people_count = 0
        unique_ids = set()
        area_of_clusters_per_frame = []
        total_confidence = 0
        average_confidence = 0
        highest_confidence = 0
        lowest_confidence = float('inf')
        dx_threshold = 8
        dy_threshold = 8
        start_time = time.time()
        threshold_exceedance_count = 0
        threshold_exceedance_frames = []
        end_time = time.time()

        is_rtsp = video_path.startswith("rtsp://")

        # For graph generation
        objects_detected_per_frame = []
        direction_counts = {'left': 0, 'right': 0, 'towards': 0, 'away': 0}
        
        # Initialize heatmap with placeholder dimensions
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA, imw=1, imh=1, view_img=False, shape="circle")
        
        # File paths for saving results
        heatmap_filename = self.heatmap_filename
        annotated_filename = self.annotated_filename
        summary_filename = self.summary_filename
        total_objects_plot_filename = self.total_objects_plot_filename
        threshold_plot_filename = self.threshold_plot_filename
        movement_plot_filename = self.movement_plot_filename

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        heatmap_out = None
        annotated_out = None
        
        # Track history for each object
        track_history = defaultdict(lambda: [])
        # Movement directions for each object
        movement_directions = defaultdict(lambda: {'left': 0, 'right': 0, 'towards': 0, 'away': 0})
        
        def process_frame(frame, new_width, new_height):
            nonlocal stop_processing, heatmap_obj, heatmap_out, annotated_out, \
                total_frames_processed, total_people_count, unique_ids, \
                area_of_clusters_per_frame, total_confidence, average_confidence, \
                highest_confidence, lowest_confidence, threshold_exceedance_count, \
                threshold_exceedance_frames, dx_threshold, dy_threshold
            
            if not is_rtsp:
                # Initialize Heatmap for non-RTSP sources with the first frame dimensions
                if self.generate_heatmap and heatmap_out is None:
                    heatmap_out = cv2.VideoWriter(heatmap_filename, fourcc, self.fps, (new_width, new_height))
                    
                # Initialize Annotated Frame for non-RTSP sources with the first frame dimensions
                if annotated_out is None:
                    annotated_out = cv2.VideoWriter(annotated_filename, fourcc, self.fps, (new_width, new_height))

                # Update heatmap dimensions if necessary for non-RTSP sources
                if self.generate_heatmap and (heatmap_obj.imw != new_width or heatmap_obj.imh != new_height):
                    heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA, imw=new_width, imh=new_height, view_img=False, shape="circle")
            
            if is_rtsp:
                # Initialize Annotated Frame for non-RTSP sources with the first frame dimensions
                if annotated_out is None:
                    annotated_out = cv2.VideoWriter(annotated_filename, fourcc, self.fps, (new_width, new_height))
                  
            # Preprocess the frame
            preprocessed_frame, _, _ = preprocess_frame(frame, self.target_width, self.target_height)
            
            # Track objects using YOLOv8, persisting tracks between frames
            if is_rtsp:
                results = self.model.track(preprocessed_frame, persist=True, device="mps", stream_buffer = True, conf = self.confidence, tracker="bytetrack.yaml")
            else:
                results = self.model.track(preprocessed_frame,  persist=True, device="mps", conf = self.confidence, tracker="bytetrack.yaml")

            if results[0].boxes is None:
                print("No objects detected in this frame.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_processing = True
                    return
                else:
                    print(f"Detected {len(results[0].boxes.xywh)} objects if boxes are not None.")

            boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            annotated_frame = frame
               
            if len(track_ids) > 0:
                annotated_frame = results[0].plot(conf = False, line_width = 2, labels = False)
                # Process each detection
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    # x, y center point
                    track.append((float(x), float(y)))
                    if len(track) > 250:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

            objects_detected_per_frame.append(len(track_ids))
            
            # Determine congestion level and suggestion & overlays congestion info on the annotated frame
            overlay_congestion_info(annotated_frame, len(track_ids), self.detection_threshold)

            # After processing detections for the frame
            if check_detection_threshold(len(track_ids), self.detection_threshold):
                threshold_exceedance_count += 1
                threshold_exceedance_frames.append(total_frames_processed)
                print(f"Threshold of {self.detection_threshold} objects exceeded in frame {total_frames_processed}.")
            
            # Calculate movement directions for each tracked object
            for track_id in track_ids:
                track = track_history[track_id]
                if len(track) >= 2:  # Making sure their is atleast two points to compare
                    prev_pos = track[-2]  # Second last element
                    current_pos = track[-1]  # Last element

                    dx = current_pos[0] - prev_pos[0]  # Change in x
                    dy = current_pos[1] - prev_pos[1]  # Change in y

                    # Apply thresholds to determine if movement is significant
                    if abs(dx) > dx_threshold:
                        if dx > 0:
                            movement_directions[track_id]['right'] += 1
                        elif dx < 0:
                            movement_directions[track_id]['left'] += 1

                    if abs(dy) > dy_threshold:
                        if dy > 0:
                            movement_directions[track_id]['towards'] += 1
                        elif dy < 0:
                            movement_directions[track_id]['away'] += 1
            
            # Update analytics            
            total_frames_processed += 1
            print(f"Successfully processed frame {total_frames_processed}...")
            total_people_count += len(track_ids)
            unique_ids.update(track_ids)
            
            # Calculate area covered by detected people
            total_area = sum([w * h for x, y, w, h in boxes])
            area_of_clusters_per_frame.append(total_area)
            
            # Update confidence metrics
            confidences = [box.conf.item() for box in results[0].boxes] if results[0].boxes else []
            total_confidence += sum(confidences)
            highest_confidence = max(highest_confidence, max(confidences)) if confidences else highest_confidence
            lowest_confidence = min(lowest_confidence, min(confidences)) if confidences else lowest_confidence

            if is_rtsp or (not is_rtsp and not self.generate_heatmap):
                cv2.imshow('Live Stream', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_processing = True
                    return
            
            if not is_rtsp:
                if self.generate_heatmap:
                    frame_with_heatmap = heatmap_obj.generate_heatmap(preprocessed_frame, results)
                    if frame_with_heatmap is not None and frame_with_heatmap.size != 0:
                        frame_with_heatmap = cv2.resize(frame_with_heatmap, (new_width, new_height))
                        heatmap_out.write(frame_with_heatmap)
                    else:
                        print("Warning: frame_with_heatmap is empty")

            # Handling annotated frames
            if annotated_frame is not None and annotated_frame.size != 0:
                if annotated_frame.shape[:2] != (new_height, new_width):
                    annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
                annotated_out.write(annotated_frame)
            else:
                print("Warning: annotated_frame is empty")

        try:
            # Check if the video path is an RTSP URL or a file path
            if is_rtsp:
                print("Attempting to pass the video information to the process_video function...")
                # Directly process the RTSP feed
                process_video(video_path, process_frame, self.fps, self.target_width, self.target_height)
                print("Finished processing the RTSP frame")
            else:
                # Ensure video file path exists for non-RTSP feeds
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                process_video(video_path, process_frame, self.fps, self.target_width, self.target_height)
                
            if stop_processing:
                return
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Processing stopped by user - detection.py")   
        except Exception as e:
            print(f"An unexpected error occurred during video processing: {e}")
            # Print the traceback to get the line number
            traceback.print_exc()
        finally:
            if heatmap_out:
                heatmap_out.release()
            if annotated_out:
                annotated_out.release()
            cv2.destroyAllWindows()
            
            frame_rate = self.fps
            total_exceedance_duration_seconds = len(threshold_exceedance_frames) / frame_rate
            
            if total_people_count > 0:
                average_confidence = float(total_confidence / total_people_count)
                lowest_confidence = float(lowest_confidence) if lowest_confidence != float('inf') else 0
                highest_confidence = float(highest_confidence)
                
            # Print analytics
            if total_frames_processed > 0:
                # End time for processing and calculate total time
                end_time = time.time()
                total_processing_time = end_time - start_time
                average_area_of_clusters = np.mean(area_of_clusters_per_frame) if area_of_clusters_per_frame else 0
                print(f"Total duration above threshold: {total_exceedance_duration_seconds} seconds")

            # Step 1: Aggregate total movements from movement_directions
            total_movements = {'left': 0, 'right': 0, 'towards': 0, 'away': 0}
            for track_id, directions in movement_directions.items():
                if isinstance(directions, dict):  # Ensure the value is a dictionary
                    for direction, count in directions.items():
                        total_movements[direction] += count

            # Step 2: Compute averages based on the total number of unique IDs
            if len(unique_ids) > 0:  # Prevents division by zero
                avg_left_movement_per_id = total_movements['left'] / len(unique_ids)
                avg_right_movement_per_id = total_movements['right'] / len(unique_ids)
                avg_towards_movement_per_id = total_movements['towards'] / len(unique_ids)
                avg_away_movement_per_id = total_movements['away'] / len(unique_ids)
            else:
                avg_left_movement_per_id = avg_right_movement_per_id = avg_towards_movement_per_id = avg_away_movement_per_id = 0

            # Determine the direction of the majority and least movement
            majority_movement = max(total_movements, key=total_movements.get)
            least_movement = min(total_movements, key=total_movements.get)
            total_processing_time = end_time - start_time
            average_area_of_clusters = np.mean(area_of_clusters_per_frame) if area_of_clusters_per_frame else 0
            
            # Directions for plotting
            for direction in direction_counts.keys():
                total_count = sum(movements.get(direction, 0) for movements in movement_directions.values())
                direction_counts[direction] = total_count
                
            # Graph for the number of objects detected per frame
            plt.figure(figsize=(10, 6))
            plt.plot(objects_detected_per_frame)
            plt.title('Number of Objects Detected Per Frame')
            plt.xlabel('Frame')
            plt.ylabel('Number of Objects')
            plt.savefig(total_objects_plot_filename)
            plt.close()

            # Graph for the number of objects detected per frame
            plt.figure(figsize=(10, 6))
            # Plot each point, change color if above threshold
            for i, count in enumerate(objects_detected_per_frame):
                color = 'red' if count >= self.detection_threshold else 'blue'
                plt.scatter(i, count, color=color)

            # Draw a horizontal line for the detection threshold
            plt.axhline(y=self.detection_threshold, color='gray', linestyle='--', label=f'Threshold ({self.detection_threshold} objects)')

            plt.title('Number of Objects Detected Per Frame')
            plt.xlabel('Frame')
            plt.ylabel('Number of Objects')
            plt.legend()

            # Save the modified graph
            plt.savefig(threshold_plot_filename)
            plt.close()

            # Graph for direction counts
            directions = list(direction_counts.keys())
            counts = [direction_counts[d] for d in directions]

            plt.figure(figsize=(10, 6))
            plt.bar(directions, counts, color=['blue', 'orange', 'green', 'red'])
            plt.title('Direction of Movement Counts')
            plt.xlabel('Direction')
            plt.ylabel('Count')
            plt.savefig(movement_plot_filename)
            plt.close()
            
            if total_frames_processed > 0:
                average_processing_time_per_frame = total_processing_time / total_frames_processed
                average_people_per_frame = total_people_count / total_frames_processed
                results_summary = (
                    "------------------------------------------\n"
                    "Summarized Results:\n"
                    f"Total processing time: {total_processing_time:.2f} seconds\n"
                    f"Total processed frames: {total_frames_processed:.2f} frames\n"
                    f"Average processing time per frame: {average_processing_time_per_frame:.2f} seconds per frame\n"
                    f"Approximate processing FPS: {(1 / average_processing_time_per_frame):.2f} fps\n"
                    f"Average number of people per frame: {average_people_per_frame:.2f}\n"
                    f"Total unique IDs detected: {len(unique_ids)}\n"
                    f"Aggregated Movements: {total_movements}\n"
                    f"Normalized Movement Directions:\n"
                    f"Left: {avg_left_movement_per_id:.2f}, Right: {avg_right_movement_per_id:.2f}, "
                    f"Towards: {avg_towards_movement_per_id:.2f}, Away: {avg_away_movement_per_id:.2f}\n"
                    f"Direction of majority of movement: {majority_movement}\n"
                    f"Direction of least movement: {least_movement}\n"
                    f"Detection Threshold: {self.detection_threshold}\n"
                    f"Number of frames threshold exceeded: {threshold_exceedance_count}\n"
                    f"Total duration above threshold: {total_exceedance_duration_seconds:.2f} seconds\n"
                    f"Average confidence level: {average_confidence:.2f}\n"
                    f"Highest confidence level: {highest_confidence:.2f}\n"
                    f"Lowest confidence level: {lowest_confidence:.2f}\n"
                    f"Average area of clusters per frame: {average_area_of_clusters:.2f}\n"
                )
            else:
                results_summary = "No frames were processed."
            
            with open(summary_filename, 'w') as file:
                file.write(results_summary)