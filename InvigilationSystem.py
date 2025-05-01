import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

#load vid
cap = cv2.VideoCapture("test.mp4")

pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

logged_flags = {"hand_raise": False, "looking_sideways": False, "mouth_movement": False}
side_look_start = None  # track sideways duration
side_look_timer_active = False

behavior_log = []

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02}:{secs:02}"

model_points = np.array([
    (0.0, 0.0, 0.0),       # nose 
    (0.0, -330.0, -65.0),  # chin
    (-225.0, 170.0, -135.0),  # left eye 
    (225.0, 170.0, -135.0),   # right eye right corner
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0)   # right mouth corner
], dtype=np.float64)

fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    video_time = current_frame / fps
    timestamp = format_timestamp(video_time)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape
    results_pose = pose.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    labels = []

    # hand raise detection
    if results_pose.pose_landmarks:
        nose_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_height
        left_wrist_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height
        right_wrist_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height

        if left_wrist_y < nose_y or right_wrist_y < nose_y:
            labels.append("Hand Raised")
            if not logged_flags["hand_raise"]:
                behavior_log.append((timestamp, "Hand raised above head"))
                logged_flags["hand_raise"] = True
        else:
            logged_flags["hand_raise"] = False

    # face mesh model
    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        # arry of face points adjusted for frame
        image_points = np.array([
            (face_landmarks[1].x * frame_width, face_landmarks[1].y * frame_height),     # nose tip
            (face_landmarks[152].x * frame_width, face_landmarks[152].y * frame_height), #chin
            (face_landmarks[263].x * frame_width, face_landmarks[263].y * frame_height), # right eye
            (face_landmarks[33].x * frame_width, face_landmarks[33].y * frame_height),   # left eye
            (face_landmarks[287].x * frame_width, face_landmarks[287].y * frame_height), #right mouth corner
            (face_landmarks[57].x * frame_width, face_landmarks[57].y * frame_height),   # left mouth corner
        ], dtype=np.float64)

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if success:
            rot_matrix, _ = cv2.Rodrigues(rvec)
            proj_matrix = cv2.hconcat((rot_matrix, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            yaw = euler_angles[1][0]
            # yaw greater than 15 start timer
            if abs(yaw) > 15:
                if not side_look_timer_active:
                    side_look_start = video_time
                    side_look_timer_active = True
                elif video_time - side_look_start > 5:
                    labels.append("Looking Sideways")
                    if not logged_flags["looking_sideways"]:
                        behavior_log.append((timestamp, f"Looking sideways (Yaw: {yaw:.2f})"))
                        logged_flags["looking_sideways"] = True
            else:
                side_look_timer_active = False
                logged_flags["looking_sideways"] = False

        # lip movement
        upper_lip_y = face_landmarks[13].y * frame_height
        lower_lip_y = face_landmarks[14].y * frame_height
        if abs(upper_lip_y - lower_lip_y) > 10:
            labels.append("Mouth Movement")
            if not logged_flags["mouth_movement"]:
                behavior_log.append((timestamp, "Mouth movement detected"))
                logged_flags["mouth_movement"] = True
        else:
            logged_flags["mouth_movement"] = False

    # frame sizing
    frame_resized = cv2.resize(frame, (640, 360))

    #frame offset
    y_offset = 30
    for label in labels:
        cv2.putText(frame_resized, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 30

    cv2.imshow("Exam Surveillance (Video)", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open("cheating_log.txt", "w") as f:
    for entry in behavior_log:
        f.write(f"{entry[0]} - {entry[1]}\n")

print("Cheating log saved to 'cheating_log.txt'")
