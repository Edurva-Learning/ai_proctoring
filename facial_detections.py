import cv2
import mediapipe as mp
import numpy as np

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


def face_mask_from_landmarks(frame, landmarks, close_kernel_size=15):
    """
    Build a robust face mask from MediaPipe face landmarks.

    This uses a convex hull over all landmark points and applies a
    morphological closing to fill small holes (glasses reflections, occlusions)
    which can otherwise create "donut" holes in masks.

    Inputs:
      - frame: BGR image (numpy array)
      - landmarks: iterable of mediapipe landmarks (landmark.x, landmark.y)
      - close_kernel_size: kernel size (odd int) for morphological closing; increase
        if you see bigger holes.

    Returns:
      - mask: single-channel uint8 mask (0 background, 255 face)
    """
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        # clamp to frame bounds
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        pts.append((x, y))

    if not pts:
        return np.zeros((h, w), dtype=np.uint8)

    pts_arr = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts_arr)

    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillConvexPoly(mask, hull, 255)
    except Exception:
        # fallback: fill polygon (works for non-convex as well)
        cv2.fillPoly(mask, [pts_arr], 255)

    # Apply morphological closing to fill small holes caused by glasses
    k = max(3, close_kernel_size // 2 * 2 + 1)  # make sure odd and >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

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


def detectFace_with_mask(frame, close_kernel_size=15):
    """Process a frame and return face count, landmark points and face masks.

    Returns: (faceCount, faces_points, masks)
      - faces_points: list of list of (x,y) tuples
      - masks: list of uint8 masks (same size as frame)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    faceCount = 0
    faces = []
    masks = []

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

            # Build a robust mask for this face
            mask = face_mask_from_landmarks(frame, face_landmarks.landmark, close_kernel_size=close_kernel_size)
            masks.append(mask)

    return (faceCount, faces, masks)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use the mask-aware detector for the demo so we can visualize
        # masks that avoid donut holes caused by glasses/occlusions.
        faceCount, faces, masks = detectFace_with_mask(frame, close_kernel_size=15)

        cv2.putText(frame, f"Faces: {faceCount}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If masks available, overlay them (semi-transparent green)
        if masks:
            overlay = frame.copy()
            for m in masks:
                colored = np.zeros_like(frame)
                colored[:, :, 1] = m  # green channel
                overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
            disp = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        else:
            disp = frame

        cv2.imshow("Face Detection (Mediapipe)", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
