import cv2
import mediapipe as mp
import numpy as np
import os

video_path = " "  # path to the videos of skateboarding tricks.
output_video_path = " "  # Path to the output files of mp4s
frame_rate = 60  # Frame rate for video
resolution = (640, 480) # Resoltion for output video

# Load the input video
cap = cv2.VideoCapture(video_path)
# Save Recorded File
out = cv2.VideoWriter("Outputvideo")
# Video is not loaded
if not cap.isOpened():
    print("Error: video cannot be opened")

    exit()

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
