import cv2

def analyze_congestion(detection_count, detection_threshold):
    congestion_levels = [
        ('Critical', {'threshold': detection_threshold * 2, 'suggestion': 'Stop new entries and consider evacuation.'}),
        ('High', {'threshold': detection_threshold * 1.5, 'suggestion': 'Reduce entry rate.'}),
        ('Medium', {'threshold': detection_threshold, 'suggestion': 'Prepare to reduce entry rate.'}),
        ('Low', {'threshold': detection_threshold * 0.5, 'suggestion': 'Monitor the situation.'}),
    ]

    for level, info in congestion_levels:
        if detection_count >= info['threshold']:
            return level, info['suggestion']
        elif detection_count < detection_threshold * 0.5:
            return 'Low', 'Monitor the situation.'

    return "Unknown", "No suggestion available."

def overlay_congestion_info(annotated_frame, detection_count, detection_threshold):
    """
    Overlay congestion level and suggestion text on the frame with a background box.
            
    :param annotated_frame: The frame on which to overlay the text.
    :param detection_count: The current count of detected objects/people.
    """
    congestion_level, suggestion = analyze_congestion(detection_count, detection_threshold)
    # Define colors for different levels
    colors = {
        'Low': (0, 255, 0),  # Green
        'Medium': (0, 255, 255),  # Yellow
        'High': (0, 165, 255),  # Orange
        'Critical': (0, 0, 255)  # Red
    }
        
     # Calculate text width and height for background box
    font_scale = 1.5
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(f"Congestion: {congestion_level}", font, font_scale, font_thickness)
        
    # Positioning the box and text from the bottom left corner
    base_line = annotated_frame.shape[0] - 10  # Start 10 pixels up from the bottom
    text_height += 20  # Add some padding
        
    # Black text color
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Draw "Detected objects:" in black
    cv2.putText(annotated_frame, "Detected objects: ", (10, base_line - 2*text_height), font, font_scale, white, font_thickness, cv2.LINE_AA)
    # Calculate the width of "Detected objects: " to position the count correctly
    (detected_objects_label_width, _), _ = cv2.getTextSize("Detected objects: ", font, font_scale, font_thickness)
    # Draw the detection count in black
    cv2.putText(annotated_frame, f"{detection_count}", (10 + detected_objects_label_width, base_line - 2*text_height), font, font_scale, white, font_thickness, cv2.LINE_AA)

     # Draw "Congestion:" in black
    cv2.putText(annotated_frame, "Congestion: ", (10, base_line - text_height), font, font_scale, white, font_thickness, cv2.LINE_AA)
     # Calculate the width of "Congestion: " to position the level correctly
    (congestion_label_width, _), _ = cv2.getTextSize("Congestion: ", font, font_scale, font_thickness)
    # Draw the congestion level in its color
    cv2.putText(annotated_frame, f"{congestion_level}", (10 + congestion_label_width, base_line - text_height), font, font_scale, colors.get(congestion_level, white), font_thickness, cv2.LINE_AA)

    # Draw "Suggestion:" and the suggestion in black
    cv2.putText(annotated_frame, "Suggestion: ", (10, base_line), font, font_scale, white, font_thickness, cv2.LINE_AA)
    # Calculate the width of "Suggestion: " to position the suggestion correctly
    (suggestion_label_width, _), _ = cv2.getTextSize("Suggestion: ", font, font_scale, font_thickness)
    cv2.putText(annotated_frame, f"{suggestion}", (10 + suggestion_label_width, base_line), font, font_scale, white, font_thickness, cv2.LINE_AA)
    
# Checks if the number of detected objects meets or exceeds the threshold.
def check_detection_threshold(num_detected, threshold):
    """
    Parameters:
    - num_detected: The number of objects detected in the current frame.
    - threshold: The threshold for the number of objects that triggers an alert. 
    """
    
    # Returns boolean indicating if the threshold is met or exceeded.
    if num_detected >= threshold:
        return True
    return False
