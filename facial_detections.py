import cv2
import mediapipe as mp

# Mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Face mesh with landmarks
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,         # detect up to 5 faces
    refine_landmarks=True,   # iris landmarks too
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detectFace(frame):
    """
    Input: Video frame (BGR)
    Output: (faceCount, faces) and draws landmarks + fancy corners
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    faceCount = 0
    faces = []

    if results.multi_face_landmarks:
        faceCount = len(results.multi_face_landmarks)

        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Collect all landmark points for this face
            face_points = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                face_points.append((x, y))

            faces.append(face_points)

            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )

            # Draw fancy corners using bounding box
            x_vals = [p[0] for p in face_points]
            y_vals = [p[1] for p in face_points]
            x, y, w_box, h_box = min(x_vals), min(y_vals), max(x_vals)-min(x_vals), max(y_vals)-min(y_vals)

            # top-left
            cv2.line(frame, (x, y), (x+20, y), (0, 255, 255), 2)
            cv2.line(frame, (x, y), (x, y+20), (0, 255, 255), 2)
            # top-right
            cv2.line(frame, (x+w_box, y), (x+w_box-20, y), (0, 255, 255), 2)
            cv2.line(frame, (x+w_box, y), (x+w_box, y+20), (0, 255, 255), 2)
            # bottom-left
            cv2.line(frame, (x, y+h_box), (x+20, y+h_box), (0, 255, 255), 2)
            cv2.line(frame, (x, y+h_box), (x, y+h_box-20), (0, 255, 255), 2)
            # bottom-right
            cv2.line(frame, (x+w_box, y+h_box), (x+w_box-20, y+h_box), (0, 255, 255), 2)
            cv2.line(frame, (x+w_box, y+h_box), (x+w_box, y+h_box-20), (0, 255, 255), 2)

    return (faceCount, faces)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faceCount, faces = detectFace(frame)

        cv2.putText(frame, f"Faces: {faceCount}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Detection (Mediapipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
