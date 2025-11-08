import cv2
import mediapipe as mp
from math import hypot

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Initialize FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmark indices (from Mediapipe FaceMesh)
# Upper lip: 13, Lower lip: 14
UPPER_LIP = 13
LOWER_LIP = 14

def calcDistance(pointA, pointB):
    return hypot((pointA[0]-pointB[0]), (pointA[1]-pointB[1]))

def mouthTrack(face_landmarks, w, h, frame):
    """Detect mouth open/close using Mediapipe landmarks"""

    upperLip = (int(face_landmarks.landmark[UPPER_LIP].x * w), 
                int(face_landmarks.landmark[UPPER_LIP].y * h))
    lowerLip = (int(face_landmarks.landmark[LOWER_LIP].x * w), 
                int(face_landmarks.landmark[LOWER_LIP].y * h))

    dist = calcDistance(upperLip, lowerLip)

    if dist > 23:   # threshold (tune if needed)
        cv2.putText(frame, "Mouth Open", (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return "Mouth Open"
    else:
        cv2.putText(frame, "Mouth Close", (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return "Mouth Close"


# ----------- MAIN PROGRAM -----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw face mesh for observation
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

            # Mouth detection
            mouthTrack(face_landmarks, w, h, frame)

    cv2.imshow("Face Observation + Mouth Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
