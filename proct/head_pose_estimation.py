import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

font = cv2.FONT_HERSHEY_PLAIN

# 3D model points for head pose estimation (same as before)
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-255.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4,1))

    rear_size, rear_depth, front_size, front_depth = val

    point_3d.extend([(-rear_size, -rear_size, rear_depth),
                     (-rear_size, rear_size, rear_depth),
                     (rear_size, rear_size, rear_depth),
                     (rear_size, -rear_size, rear_depth),
                     (-rear_size, -rear_size, rear_depth)])

    point_3d.extend([(-front_size, -front_size, front_depth),
                     (-front_size, front_size, front_depth),
                     (front_size, front_size, front_depth),
                     (front_size, -front_size, front_depth),
                     (-front_size, -front_size, front_depth)])

    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)
    point_2d, _ = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    return np.int32(point_2d.reshape(-1, 2))

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]
    return (x, y)

def head_pose_detection(img):
    size = img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Map MediaPipe landmarks to the points used before
            image_points = np.array([
                [face_landmarks.landmark[1].x * size[1], face_landmarks.landmark[1].y * size[0]],  # Nose tip
                [face_landmarks.landmark[152].x * size[1], face_landmarks.landmark[152].y * size[0]],  # Chin
                [face_landmarks.landmark[263].x * size[1], face_landmarks.landmark[263].y * size[0]],  # Right eye
                [face_landmarks.landmark[33].x * size[1], face_landmarks.landmark[33].y * size[0]],    # Left eye
                [face_landmarks.landmark[287].x * size[1], face_landmarks.landmark[287].y * size[0]],  # Right mouth
                [face_landmarks.landmark[57].x * size[1], face_landmarks.landmark[57].y * size[0]],    # Left mouth
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
            )

            # Project nose point
            nose_end_point2D, _ = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            # Head angles
            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

            if ang1 >= 45:
                cv2.putText(img, 'Head up', (50, 50), font, 2, (255, 255, 128), 2)
                return "Head Up"
            elif ang1 <= -45:
                cv2.putText(img, 'Head down', (50, 50), font, 2, (255, 255, 128), 2)
                return "Head Down"
            if ang2 >= 45:
                cv2.putText(img, 'Head right', (50, 30), font, 2, (255, 255, 128), 2)
                return "Head Right"
            elif ang2 <= -45:
                cv2.putText(img, 'Head left', (50, 30), font, 2, (255, 255, 128), 2)
                return "Head Left"

    return -1

# Example usage
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break
    head_pose_detection(img)
    cv2.imshow("Head Pose", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
