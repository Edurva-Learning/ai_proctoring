import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # gives iris landmarks too
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Left & Right Eye landmark indices (Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]  
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eyeSegmentationAndReturnWhite(img, side):
    """Count white pixels in left or right half of eye image"""
    height, width = img.shape
    if side == "left":
        img = img[:, :width//2]
    else:
        img = img[:, width//2:]
    return cv2.countNonZero(img)

def gazeDetection(frame):
    """Detect gaze direction using Mediapipe FaceMesh"""
    result_text = ""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Collect eye regions
            leftEyeRegion = np.array([(int(face_landmarks.landmark[i].x * w),
                                       int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE], np.int32)
            rightEyeRegion = np.array([(int(face_landmarks.landmark[i].x * w),
                                        int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE], np.int32)

            # Create mask
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [leftEyeRegion], 255)
            cv2.fillPoly(mask, [rightEyeRegion], 255)

            eyes = cv2.bitwise_and(frame, frame, mask=mask)

            # Crop eye regions
            lmin_x, lmax_x = np.min(leftEyeRegion[:, 0]), np.max(leftEyeRegion[:, 0])
            lmin_y, lmax_y = np.min(leftEyeRegion[:, 1]), np.max(leftEyeRegion[:, 1])
            rmin_x, rmax_x = np.min(rightEyeRegion[:, 0]), np.max(rightEyeRegion[:, 0])
            rmin_y, rmax_y = np.min(rightEyeRegion[:, 1]), np.max(rightEyeRegion[:, 1])

            leftEye = eyes[lmin_y:lmax_y, lmin_x:lmax_x]
            rightEye = eyes[rmin_y:rmax_y, rmin_x:rmax_x]

            # Convert to grayscale
            leftGray = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
            rightGray = cv2.cvtColor(rightEye, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            leftTh = cv2.adaptiveThreshold(leftGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            rightTh = cv2.adaptiveThreshold(rightGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # Count white pixels
            leftSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, "left")
            rightSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, "right")
            leftSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, "left")
            rightSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, "right")

            # Gaze decision
            TrialRation = 1.2
            if rightSideOfRightEye >= TrialRation * leftSideOfRightEye:
                result_text = "Looking Left"
            elif leftSideOfLeftEye >= TrialRation * rightSideOfLeftEye:
                result_text = "Looking Right"
            else:
                result_text = "Looking Center"

            # Show text on frame
            cv2.putText(frame, result_text, (50, 110),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)

    return result_text


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        direction = gazeDetection(frame)
        print(direction)

        cv2.imshow("Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
