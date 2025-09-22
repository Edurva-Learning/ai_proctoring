import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices (Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(pt1, pt2):
    return math.dist(pt1, pt2)

def eye_aspect_ratio(eye_points, landmarks, w, h):
    p1 = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
    p2 = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
    p3 = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))
    p4 = (int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h))
    p5 = (int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h))
    p6 = (int(landmarks[eye_points[5]].x * w), int(landmarks[eye_points[5]].y * h))

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def isBlinking(frame):
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(LEFT_EYE, face_landmarks.landmark, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, face_landmarks.landmark, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.21:
                return (left_ear, right_ear, "Blink")
            else:
                return (left_ear, right_ear, "Open")
    return (0, 0, "No Face")

# ---------------- Webcam Loop ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left_ear, right_ear, status = isBlinking(frame)

    # Display status on frame
    cv2.putText(frame, f"{status} | L: {left_ear:.2f} R: {right_ear:.2f}", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Blink Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
