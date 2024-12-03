import cv2
import mediapipe as mp
import numpy as np
import os

# Video is not loaded
# Video is not loaded
video_path = "./data/skate.mp4"  # path to the videos of skateboarding tricks.
output_video_path = "./output/output.mp4"  # Path to the output files of mp4s
frame_rate = 60  # Frame rate for video
resolution = (640, 480)  # Resoltion for output video
window_name = "Trick"

cv2.namedWindow(window_name)
# Load the input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():  # Check if video can be loaded
    print("Error: video cannot be opened")
    exit()

original_fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_resolution = (original_width, original_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, original_fps, output_resolution)

# Process video frames
while cap.isOpened():  # Capture each frame of the video

    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read frame.")
        break

    # Resize frame if resolution is specified
    if resolution:
        frame = cv2.resize(frame, resolution)

    # Example processing: Convert frame to grayscale
    processed_frame = frame

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Display the processed frame
    cv2.imshow("Trick", processed_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
