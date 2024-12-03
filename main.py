import cv2
import mediapipe as mp
import numpy as np
import os

# Video is not loaded
# Video is not loaded
video_path = "./data/skate.mp4"  # path to the videos of skateboarding tricks.
output_video_path = " .oout"  # Path to the output files of mp4s
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
out = out = cv2.VideoWriter(output_video_path, fourcc, original_fps, output_resolution)

while cap.isOpened():  # Capture each frame of the video
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        else:
            break

cap.release()
cv2.destroyAllWindows()
