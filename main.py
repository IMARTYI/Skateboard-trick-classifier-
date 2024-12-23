import cv2
import mediapipe as mp
import os
# Video paths
video_path = "./data"  # Path to the input video
window_name = "Trick"
csv_results = []  # List to output csv results
frame_number = 0
output_video_path = "./output"  # Path to the output video
# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load input video

def write_landmarks_to_csv(csv_data, frame_number, landmarks):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

for filename in os.listdir(video_path):

    path = os.path.join(video_path, filename)
    cap = cv2.VideoCapture(path)

    output_path = os.path.join(output_video_path,filename)
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
    out = cv2.VideoWriter(output_path, fourcc, original_fps, output_resolution)

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

            write_landmarks_to_csv(csv_results, frame_number, results.pose_landmarks.landmark)

        # Write the processed frame to the output video
        out.write(frame)
        # Display the frame with landmarks
        cv2.imshow(window_name, frame)
        frame_number += 1  # Increment frame number
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Function that will out put each landmark coordinates

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
