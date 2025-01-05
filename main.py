import csv
import cv2
import mediapipe as mp
import os

# Paths
video_path = "./data"  # Path to the input videos
output_video_path = "./output"  # Path to the output video
output_csv_path = "./landmarks_filtered.csv"  # Path to save the filtered CSV file

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Relevant landmarks (feet, legs, upper body, arms)
RELEVANT_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

# Create CSV file and write header
with open(output_csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header
    header = ["Frame"]
    for landmark in RELEVANT_LANDMARKS:
        header.extend([
            f"{landmark.name}_x",
            f"{landmark.name}_y",
            f"{landmark.name}_z"
        ])
    csv_writer.writerow(header)


    for filename in os.listdir(video_path):
        path = os.path.join(video_path, filename)
        cap = cv2.VideoCapture(path)

        output_path = os.path.join(output_video_path, filename)
        if not cap.isOpened():
            print(f"Error: Unable to open video {filename}")
            continue

        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_resolution = (original_width, original_height)

        # Set up output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, output_resolution)

        frame_number = 0
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video {filename} reached or cannot read frame.")
                break

            # Convert frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)


            if results.pose_landmarks:
                row = [frame_number]
                for landmark in RELEVANT_LANDMARKS:
                    data = results.pose_landmarks.landmark[landmark]
                    row.extend([data.x, data.y, data.z])
                csv_writer.writerow(row)

                # Draw the full skeleton pose
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Emphasize relevant landmarks with larger circles
                for landmark in RELEVANT_LANDMARKS:
                    landmark_data = results.pose_landmarks.landmark[landmark]
                    cx, cy = int(landmark_data.x * original_width), int(landmark_data.y * original_height)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)  # Red circles for relevant landmarks

            # Write the processed frame to the output video
            out.write(frame)

            # Display the frame with landmarks
            cv2.imshow("Trick", frame)
            frame_number += 1

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources for this video
        cap.release()
        out.release()

# Release all resources
cv2.destroyAllWindows()
pose.close()
