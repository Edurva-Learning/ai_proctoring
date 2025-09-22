import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import winsound
import pyaudio
import struct

# Mediapipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Global records
data_record = []
running = True

# Beep settings
frequency = 2500
duration = 1000

# EAR threshold
EAR_THRESHOLD = 0.2

# Eye landmarks (MediaPipe indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
UPPER_LIP = 13
LOWER_LIP = 14

# Audio detection settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AUDIO_THRESHOLD = 500  # Adjust this threshold based on testing

audio = pyaudio.PyAudio()

# ----------------- FUNCTIONS -----------------

def eye_aspect_ratio(landmarks, eye_points):
    p1 = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    p2 = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])
    p3 = np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    p4 = np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])

    p_left = np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y])
    p_right = np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])

    dist_v1 = np.linalg.norm(p2 - p4)
    dist_v2 = np.linalg.norm(p1 - p3)
    dist_h = np.linalg.norm(p_left - p_right)

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear


def isBlinking(landmarks):
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    return "Blink" if ear < EAR_THRESHOLD else "Open"


def mouthTrack(landmarks):
    upper_lip = np.array([landmarks[UPPER_LIP].x, landmarks[UPPER_LIP].y])
    lower_lip = np.array([landmarks[LOWER_LIP].x, landmarks[LOWER_LIP].y])
    dist = np.linalg.norm(upper_lip - lower_lip)
    return "Mouth Open" if dist > 0.04 else "Mouth Closed"


def gazeDetection(landmarks):
    left_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] for p in LEFT_EYE], axis=0)
    right_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] for p in RIGHT_EYE], axis=0)
    avg_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    if avg_x < 0.4:
        return "Looking Left"
    elif avg_x > 0.6:
        return "Looking Right"
    else:
        return "Looking Center"


def head_pose_detection(landmarks):
    nose = np.array([landmarks[1].x, landmarks[1].y])
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    eye_center = (left_eye + right_eye) / 2
    dx = nose[0] - eye_center[0]
    if dx > 0.05:
        return "Head Right"
    elif dx < -0.05:
        return "Head Left"
    else:
        return "Head Center"


def detectObject(frame):
    # Placeholder for YOLO/SSD detection
    return ["Person"]


def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(2)
        winsound.Beep(frequency, duration)
        return "Multiple faces detected"
    elif faceCount == 0:
        time.sleep(2)
        winsound.Beep(frequency, duration)
        return "No face detected"
    else:
        return "Face detected"


def audio_detection(stream):
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype=np.int16)
    volume = np.abs(audio_data).mean()
    return "Speaking" if volume > AUDIO_THRESHOLD else "Silence"


# ----------------- MAIN LOOP -----------------

def proctoringAlgo():
    cam = cv2.VideoCapture(0)
    blinkCount = 0

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while running:
            ret, frame = cam.read()
            if not ret:
                break

            record = []
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            record.append(current_time)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Audio Detection
            audio_status = audio_detection(stream)
            record.append(audio_status)
            print("Audio Status:", audio_status)

            if results.multi_face_landmarks:
                faceCount = len(results.multi_face_landmarks)
                remark = faceCount_detection(faceCount)
                record.append(remark)
                print(remark)

                for landmarks in results.multi_face_landmarks:
                    # Blink
                    blinkStatus = isBlinking(landmarks.landmark)
                    if blinkStatus == "Blink":
                        blinkCount += 1
                        record.append(f"Blink count: {blinkCount}")
                    else:
                        record.append(blinkStatus)
                    print(blinkStatus)

                    # Gaze
                    gaze = gazeDetection(landmarks.landmark)
                    record.append(gaze)
                    print(gaze)

                    # Mouth
                    mouth_status = mouthTrack(landmarks.landmark)
                    record.append(mouth_status)
                    print(mouth_status)

                    # Object detection
                    objects = detectObject(frame)
                    record.append(objects)
                    print("Objects:", objects)

                    # Head pose
                    head_pose = head_pose_detection(landmarks.landmark)
                    record.append(head_pose)
                    print(head_pose)

                    # Draw mesh
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

            else:
                record.append("No face detected")

            data_record.append(record)
            cv2.imshow("Proctoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    stream.stop_stream()
    stream.close()
    audio.terminate()
    cam.release()
    cv2.destroyAllWindows()


def main_app():
    activityVal = "\n".join(map(str, data_record))
    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))


if __name__ == "__main__":
    proctoringAlgo()
    main_app()
