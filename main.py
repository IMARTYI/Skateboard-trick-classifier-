import cv2
import mediapipe as mp
import os

# Video paths
video_path = "./data/skate.mp4"  # Path to the input video
output_video_path = "./output/output.mp4"  # Path to the output video
window_name = "Trick"

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: video cannot be opened")
    exit()

# Get video properties
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_resolution = (original_width, original_height)

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, original_fps, output_resolution)

cv2.namedWindow(window_name)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read frame.")
        break

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with landmarks
    cv2.imshow(window_name, frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
