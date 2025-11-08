import cv2
import mediapipe as mp
import asyncio
import base64
import numpy as np
import time
import struct
from datetime import datetime
from typing import Dict

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
# ======= Socket.IO server setup =======
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(title="AI Proctoring Server")
 
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.68.110:3000", "http://192.168.68.108:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
asgi_app = socketio.ASGIApp(sio, app)
 
# ======= Mediapipe setup =======
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------- Detection helper settings (ported from temp.py) ----------------
EAR_THRESHOLD = 0.2
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14

# Audio detection threshold (for base64 PCM16 audio blobs)
AUDIO_THRESHOLD = 500

# Try to load YOLO model (yolov8n.pt) if ultralytics is installed and model file exists
try:
    from ultralytics import YOLO
    try:
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded")
    except Exception as e:
        yolo_model = None
        print(f"⚠️ YOLO model failed to load: {e}")
except Exception:
    YOLO = None
    yolo_model = None
    print("⚠️ ultralytics.YOLO not available; object detection disabled")

# Control whether detected object labels/confidences are drawn on frames.
# Set to False to avoid showing object names in the video overlay (recommended for privacy/UI clarity).
SHOW_OBJECT_LABELS = False


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

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h) if dist_h != 0 else 0.0
    return ear


def is_blinking(landmarks):
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    return "Blink" if ear < EAR_THRESHOLD else "Open"


def mouth_track(landmarks):
    upper_lip = np.array([landmarks[UPPER_LIP].x, landmarks[UPPER_LIP].y])
    lower_lip = np.array([landmarks[LOWER_LIP].x, landmarks[LOWER_LIP].y])
    dist = np.linalg.norm(upper_lip - lower_lip)
    return "Mouth Open" if dist > 0.04 else "Mouth Closed"


def gaze_detection(landmarks):
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
    # simple left/right heuristic using nose vs eyes
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


def detect_object(frame, draw_labels: bool = False):
    # returns list of labels or ['No object']
    if yolo_model is None:
        return []
    try:
        results = yolo_model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # draw box for visualization only when requested
                if draw_labels or SHOW_OBJECT_LABELS:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detections.append({
                    'label': label,
                    'conf': conf,
                    'box': (x1, y1, x2, y2)
                })
        return detections
    except Exception as e:
        print(f"⚠️ YOLO detection error: {e}")
        return []


def audio_detection_from_b64(b64_audio: str):
    # Expects base64-encoded raw PCM16 (s16le) audio or a WAV payload; we'll try raw decode
    try:
        if b64_audio.startswith('data:'):
            b64_audio = b64_audio.split(',')[1]
        audio_bytes = base64.b64decode(b64_audio)
        # Try unpacking as int16 PCM
        count = len(audio_bytes) // 2
        if count <= 0:
            return "Silence"
        fmt = f"{count}h"
        samples = struct.unpack(fmt, audio_bytes[: count * 2])
        audio_data = np.array(samples, dtype=np.int16)
        volume = np.abs(audio_data).mean()
        return "Speaking" if volume > AUDIO_THRESHOLD else "Silence"
    except Exception as e:
        print(f"⚠️ Audio detection failed: {e}")
        return "Unknown"
 
# ======= In-Memory Storage for Proctoring Sessions =======
# store session keys as strings to avoid int/str mismatches from client
active_sessions: Dict[str, Dict] = {}
 
# ======= Socket.IO events =======
@sio.event
async def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
 
@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")
    # Clean up sessions
    for session_id, session_data in list(active_sessions.items()):
        if sid in session_data.get('faculty_members', []):
            session_data['faculty_members'].remove(sid)
            print(f"Removed faculty {sid} from session {session_id}")
        # If a student socket disconnected, clear the socket id but keep the
        # registered student_id so future reconnects with the same userId
        # will be matched to this session (prevents accidental duplicate sessions).
        if sid == session_data.get('student_sid'):
            active_sessions[session_id]['student_sid'] = None
            # DO NOT clear student_id here; keep association to allow reuse.
            print(f"Cleared student SID {sid} from session {session_id} (student_id retained)")
 
@sio.on("join_room")
async def on_join_room(sid, data):
    """Universal room joining for both faculty and students"""
    # Normalize incoming IDs to strings to avoid int/str key mismatches
    session_id_raw = data.get("sessionId")
    user_type = data.get("userType")  # 'faculty' or 'student'
    user_id_raw = data.get("userId")

    # Convert to strings when present
    session_id = str(session_id_raw) if session_id_raw is not None else None
    user_id = str(user_id_raw) if user_id_raw is not None else None

    print(f"Received join_room: sessionId={session_id}, userType={user_type}, userId={user_id}")

    if not session_id and not user_id:
        await sio.emit("error", {"message": "Session ID or user ID required"}, to=sid)
        return

    # If student is joining and they already have a session, reuse it instead of creating a new one
    if user_type == 'student' and user_id:
        existing_session = None
        for s_id, s_data in active_sessions.items():
            if s_data.get('student_id') == user_id:
                existing_session = s_id
                break
        if existing_session and existing_session != session_id:
            # Join the existing session and update SID
            room_name = f"exam_{existing_session}"
            sio.enter_room(sid, room_name)
            active_sessions[existing_session]['student_sid'] = sid
            active_sessions[existing_session]['student_id'] = user_id
            print(f"🔁 Reusing existing proctoring session {existing_session} for student {user_id}")
            await sio.emit("room_joined", {
                "room": room_name,
                "sessionId": existing_session,
                "userType": user_type,
                "sessionStatus": "active"
            }, to=sid)

            # Notify faculty about this (if any)
            if active_sessions[existing_session]['faculty_members']:
                await sio.emit("student_joined", {"userId": user_id, "sessionId": existing_session}, room=room_name)
            return

    room_name = f"exam_{session_id or ''}"

    # Enter the room
    sio.enter_room(sid, room_name)
    print(f"👥 {user_type.capitalize()} {user_id or sid} joined room {room_name}")
   
    # Initialize session if not exists (use string keys)
    if session_id and session_id not in active_sessions:
        active_sessions[session_id] = {
            'faculty_members': [],
            'student_sid': None,
            'student_id': None,
            'last_frame': None,
            'last_detection': None,
            'blink_count': 0,
            'audio_events': [],
            'object_history': [],
            'created_at': datetime.now().isoformat()
        }
        print(f"🏠 Created new proctoring session {session_id}")
   
    # Update session based on user type
    if user_type == 'faculty':
        if session_id and sid not in active_sessions[session_id]['faculty_members']:
            active_sessions[session_id]['faculty_members'].append(sid)
    elif user_type == 'student':
        # assign student info (ensure session exists)
        if session_id not in active_sessions:
            # fallback: create session for this student
            active_sessions[session_id] = {
                'faculty_members': [],
                'student_sid': sid,
                'student_id': user_id,
                'last_frame': None,
                'last_detection': None,
                'blink_count': 0,
                'audio_events': [],
                'object_history': [],
                'created_at': datetime.now().isoformat()
            }
            print(f"🏠 Created new proctoring session {session_id} (late student join)")
        else:
            active_sessions[session_id]['student_sid'] = sid
            active_sessions[session_id]['student_id'] = user_id
   
    # Send confirmation
    await sio.emit("room_joined", {
        "room": room_name,
        "sessionId": session_id,
        "userType": user_type,
        "sessionStatus": "active" if active_sessions[session_id]['student_sid'] else "waiting"
    }, to=sid)
   
    # Notify faculty about new student
    if user_type == 'student' and active_sessions[session_id]['faculty_members']:
        await sio.emit("student_joined", {
            "userId": user_id,
            "sessionId": session_id
        }, room=room_name)
 
@sio.on("stream_frame")
async def on_stream_frame(sid, data):
    """Student sends frames for AI proctoring processing"""
    # Normalize incoming IDs to strings
    session_id_raw = data.get("sessionId")
    user_id_raw = data.get("userId")
    frame_data = data.get("frame")

    session_id = str(session_id_raw) if session_id_raw is not None else None
    user_id = str(user_id_raw) if user_id_raw is not None else None

    if not all([session_id, user_id, frame_data]):
        print(f"❌ Missing data in stream_frame from {sid}")
        return
 
    # Check if session exists and student is registered
    # If the provided session_id does not exist, try to find an existing
    # session for this student by student_id and reuse it to avoid duplicates.
    if session_id not in active_sessions:
        existing_session = None
        for s_id, s_data in active_sessions.items():
            if s_data.get('student_id') == user_id:
                existing_session = s_id
                break
        if existing_session:
            print(f"🔁 Reusing existing session {existing_session} for stream from student {user_id}")
            session_id = existing_session
        else:
            print(f"❌ Session {session_id} does not exist for stream from {user_id}")
            await sio.emit("error", {"message": f"Session {session_id} does not exist"}, to=sid)
            return
   
    # If student hasn't been registered yet (join_room missed), accept first streaming SID
    if active_sessions[session_id].get('student_sid') is None:
        active_sessions[session_id]['student_sid'] = sid
        active_sessions[session_id]['student_id'] = user_id
        room_name = f"exam_{session_id}"
        try:
            await sio.emit("student_joined", {"userId": user_id, "sessionId": session_id}, room=room_name)
        except Exception as e:
            print(f"⚠️ Failed to notify faculty of late student join: {e}")
        print(f"✅ Registered student SID from first frame for session {session_id}")
    elif active_sessions[session_id].get('student_sid') != sid:
        # Different socket trying to stream and a student already registered
        print(f"❌ Unauthorized stream attempt from {sid} for session {session_id}")
        return
 
    try:
        # Decode base64 image
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
       
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if frame is None:
            print("❌ Failed to decode frame")
            return
 
        # Extract optional audio blob (base64) if client sent it
        audio_blob = data.get('audio')
        audio_status = None
        if audio_blob:
            audio_status = audio_detection_from_b64(audio_blob)
            print("Audio Status:", audio_status)
            # store short history
            active_sessions[session_id].setdefault('audio_events', []).append({
                'timestamp': datetime.now().isoformat(), 'status': audio_status
            })

        # Process frame with AI proctoring (face detection, eye tracking, head pose)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
           
            # Minimal detection data with proctoring alerts only
            detection_data = {
                "userId": user_id,
                "sessionId": session_id,
                "faceDetected": False,
                "multipleFaces": False,
                "timestamp": datetime.now().isoformat(),
                "landmarksCount": 0,
                "proctoringAlerts": []
            }
 
            if results.multi_face_landmarks:
                detection_data["faceDetected"] = True
                detection_data["landmarksCount"] = len(results.multi_face_landmarks[0].landmark)
               
                # Check for multiple faces (cheating alert)
                if len(results.multi_face_landmarks) > 1:
                    detection_data["multipleFaces"] = True
                    detection_data["proctoringAlerts"].append("Multiple faces detected")
               
                # Draw landmarks on frame for visualization and compute per-face metrics
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )
                    except Exception as draw_error:
                        # Fallback to simple drawing
                        print(f"⚠️ Drawing styles error, using fallback: {draw_error}")
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )

                    # compute detection metrics (use landmarks list)
                    try:
                        landmarks = face_landmarks.landmark

                        # Blink -> append only when blink detected (counts maintained per session)
                        blink_status = is_blinking(landmarks)
                        if blink_status == "Blink":
                            active_sessions[session_id]['blink_count'] = active_sessions[session_id].get('blink_count', 0) + 1
                            blink_count = active_sessions[session_id]['blink_count']
                            detection_data['proctoringAlerts'].append(f"Blink count: {blink_count}")
                            print(f"Blink count: {blink_count}")

                        # Audio: already processed above; alert on speaking
                        if audio_status == "Speaking":
                            detection_data['proctoringAlerts'].append("Speaking detected")
                            print("Speaking detected")

                        # Gaze -> alert when not centered
                        gaze = gaze_detection(landmarks)
                        if gaze != "Looking Center":
                            detection_data['proctoringAlerts'].append(gaze)
                            print(gaze)

                        # Mouth -> alert when open
                        mouth_status = mouth_track(landmarks)
                        if mouth_status == "Mouth Open":
                            detection_data['proctoringAlerts'].append(mouth_status)
                            print(mouth_status)

                        # Head pose -> alert when turned left/right
                        head_pose = head_pose_detection(landmarks)
                        if head_pose != "Head Center":
                            detection_data['proctoringAlerts'].append(head_pose)
                            print(head_pose)

                        # Object detection (YOLO) - operate on the current frame image
                        objects = detect_object(frame)
                        # Only alert when meaningful objects are detected
                        if objects and not any(x in objects for x in ("No object", "Object detection unavailable", "Detection error")):
                            # Normalize labels for matching
                            labels_lower = [str(o).lower() for o in objects]

                            # objects is now a list of detection dicts
                            # Normalize labels and extract boxes
                            labels_lower = [str(d['label']).lower() for d in objects]
                            boxes = [d.get('box') for d in objects]

                            # Device keywords to alert on (handle common label variants)
                            device_keywords = ['cell', 'phone', 'mobile', 'laptop', 'notebook', 'monitor', 'screen', 'tv', 'television']

                            # Build face bounding box (pixels) from landmarks to ignore object detections overlapping the face
                            try:
                                xs = [int(lm.x * frame_w) for lm in landmarks]
                                ys = [int(lm.y * frame_h) for lm in landmarks]
                                fx1, fy1 = max(0, min(xs)), max(0, min(ys))
                                fx2, fy2 = min(frame_w - 1, max(xs)), min(frame_h - 1, max(ys))
                                pad = int(min(frame_w, frame_h) * 0.05)
                                fx1, fy1 = max(0, fx1 - pad), max(0, fy1 - pad)
                                fx2, fy2 = min(frame_w - 1, fx2 + pad), min(frame_h - 1, fy2 + pad)
                                face_box = (fx1, fy1, fx2, fy2)
                            except Exception:
                                face_box = None

                            # Device detection filtering: require higher confidence and ignore detections overlapping the face
                            min_conf_device = 0.50
                            devices_found = []
                            for d in objects:
                                lbl = str(d.get('label', '')).lower()
                                conf = float(d.get('conf', 0.0) or 0.0)
                                box = d.get('box')
                                if not box:
                                    continue
                                if any(k in lbl for k in device_keywords) and conf >= min_conf_device:
                                    # ignore if device box overlaps significantly with face box (likely misclassification on face)
                                    try:
                                        if face_box is not None:
                                            if iou(face_box, box) > 0.25:
                                                # overlapping with face -> ignore
                                                continue
                                    except Exception:
                                        pass
                                    devices_found.append(d['label'])
                            # dedupe preserving order
                            devices_found = list(dict.fromkeys(devices_found))

                            # Strict person counting with spatial deduplication and filters
                            # Only consider person detections above a confidence and minimum area to avoid tiny/false boxes
                            frame_h, frame_w = frame.shape[:2]
                            frame_area = max(1, frame_w * frame_h)
                            min_conf = 0.45
                            min_area_ratio = 0.01  # at least 1% of frame area

                            person_boxes = []
                            for d in objects:
                                lbl = str(d.get('label', '')).lower()
                                conf = float(d.get('conf', 0.0) or 0.0)
                                box = d.get('box')
                                if box is None:
                                    continue
                                x1, y1, x2, y2 = box
                                w = max(0, x2 - x1)
                                h = max(0, y2 - y1)
                                area = w * h
                                if 'person' in lbl and conf >= min_conf and area >= (min_area_ratio * frame_area):
                                    person_boxes.append(box)

                            def iou(a, b):
                                (x1, y1, x2, y2) = a
                                (x3, y3, x4, y4) = b
                                xi1 = max(x1, x3)
                                yi1 = max(y1, y3)
                                xi2 = min(x2, x4)
                                yi2 = min(y2, y4)
                                inter_w = max(0, xi2 - xi1)
                                inter_h = max(0, yi2 - yi1)
                                inter = inter_w * inter_h
                                area_a = max(0, x2 - x1) * max(0, y2 - y1)
                                area_b = max(0, x4 - x3) * max(0, y4 - y3)
                                union = area_a + area_b - inter
                                return inter / union if union > 0 else 0

                            # cluster person boxes: consider boxes with IoU > 0.75 as same person (strict)
                            clusters = []
                            iou_threshold = 0.80
                            for pb in person_boxes:
                                placed = False
                                for c in clusters:
                                    if any(iou(pb, other) > iou_threshold for other in c):
                                        c.append(pb)
                                        placed = True
                                        break
                                if not placed:
                                    clusters.append([pb])

                            # Merge clusters whose centers are very close (to avoid counting jittery nearby boxes as separate persons)
                            def center_of(box):
                                x1, y1, x2, y2 = box
                                return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                            merged_boxes = []
                            for c in clusters:
                                xs = [b[0] for b in c]
                                ys = [b[1] for b in c]
                                x2s = [b[2] for b in c]
                                y2s = [b[3] for b in c]
                                merged_boxes.append((min(xs), min(ys), max(x2s), max(y2s)))

                            # center distance threshold (fraction of frame diagonal)
                            diag = (frame_w**2 + frame_h**2) ** 0.5
                            center_thresh = diag * 0.12  # centers closer than 12% of diag -> merge

                            final_boxes = []
                            for mb in merged_boxes:
                                mc = center_of(mb)
                                merged_into_existing = False
                                for i, fb in enumerate(final_boxes):
                                    fc = center_of(fb)
                                    dist = ((mc[0] - fc[0])**2 + (mc[1] - fc[1])**2) ** 0.5
                                    if dist < center_thresh:
                                        # merge boxes
                                        nx1 = min(fb[0], mb[0])
                                        ny1 = min(fb[1], mb[1])
                                        nx2 = max(fb[2], mb[2])
                                        ny2 = max(fb[3], mb[3])
                                        final_boxes[i] = (nx1, ny1, nx2, ny2)
                                        merged_into_existing = True
                                        break
                                if not merged_into_existing:
                                    final_boxes.append(mb)

                            # Final person count after deduplication
                            person_count = len(final_boxes)

                            # Alert only for multiple persons or specific devices
                            alerted = False
                            if person_count > 1:
                                detection_data['proctoringAlerts'].append("Multiple persons detected")
                                alerted = True
                                print("Multiple persons detected")

                            if devices_found:
                                objs_str = ", ".join(devices_found)
                                detection_data['proctoringAlerts'].append(f"Prohibited objects detected: {objs_str}")
                                # record object history in session
                                active_sessions[session_id].setdefault('object_history', []).append({
                                    'timestamp': datetime.now().isoformat(), 'objects': devices_found
                                })
                                alerted = True
                                print("Prohibited objects:", devices_found)

                            # If only a single person was detected and no targeted device, do not alert
                            if not alerted:
                                # No alert-worthy objects found (e.g., single person only)
                                pass
                    except Exception as det_err:
                        print(f"⚠️ Detection metric error: {det_err}")
 
            # Add no-face detection alert
            if not detection_data["faceDetected"]:
                detection_data["proctoringAlerts"].append("No face detected")
 
            # Convert processed frame back to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_data_url = f"data:image/jpeg;base64,{frame_b64}"
 
            # Store latest frame and detection data
            active_sessions[session_id]['last_frame'] = frame_data_url
            active_sessions[session_id]['last_detection'] = detection_data
 
            # Prepare payload for faculty monitor
            frame_data_payload = {
                "userId": user_id,
                "sessionId": session_id,
                "frame": frame_data_url,
                "detection": detection_data
            }
           
            # Emit to the exam room (faculty monitor)
            room_name = f"exam_{session_id}"
            await sio.emit("receive_frame", frame_data_payload, room=room_name)
 
            # Additional direct emits to registered faculty SIDs as a robustness fallback
            faculty_list = active_sessions.get(session_id, {}).get('faculty_members', []) or []
            for fac_sid in faculty_list:
                try:
                    await sio.emit("receive_frame", frame_data_payload, to=fac_sid)
                except Exception as e:
                    print(f"⚠️ Failed to emit frame directly to faculty {fac_sid}: {e}")
 
            print(f"📹 Frame sent to room {room_name} (faculty: {len(faculty_list)}) - Face: {detection_data['faceDetected']} - Alerts: {len(detection_data['proctoringAlerts'])}")
 
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        await sio.emit("error", {"message": f"Frame processing error: {str(e)}"}, to=sid)
 
@sio.on("get_session_status")
async def on_get_session_status(sid, data):
    """Get status of a proctoring session"""
    session_id_raw = data.get("sessionId")
    if not session_id_raw:
        await sio.emit("error", {"message": "Session ID required"}, to=sid)
        return

    session_id = str(session_id_raw)

    if session_id in active_sessions:
        session = active_sessions[session_id]
        last_det = session.get('last_detection') or {}
        status = {
            "sessionId": session_id,
            "hasStudent": session.get('student_sid') is not None,
            "studentId": session.get('student_id'),
            "facultyCount": len(session.get('faculty_members', [])),
            "lastUpdate": last_det.get('timestamp'),
            "faceDetected": last_det.get('faceDetected', False),
            "alerts": last_det.get('proctoringAlerts', [])
        }
        await sio.emit("session_status", status, to=sid)
    else:
        await sio.emit("session_status", {
            "sessionId": session_id,
            "exists": False,
            "message": "Session not found"
        }, to=sid)
 
@sio.on("end_proctoring")
async def on_end_proctoring(sid, data):
    """End proctoring session"""
    session_id_raw = data.get("sessionId")
    if not session_id_raw:
        return

    session_id = str(session_id_raw)

    if session_id in active_sessions:
        # Notify all participants
        room_name = f"exam_{session_id}"
        await sio.emit("proctoring_ended", {
            "sessionId": session_id,
            "message": "Proctoring session ended"
        }, room=room_name)
       
        # Clean up session
        del active_sessions[session_id]
        print(f"🗑️ Proctoring session {session_id} ended and cleaned up")
 
# ======= Health Check Endpoints =======
@app.get("/session/{session_id}/status")
async def get_session_status(session_id: int):
    """Get session status via API"""
    # active_sessions uses string keys, convert the path param to string
    session_key = str(session_id)
    if session_key in active_sessions:
        session = active_sessions[session_key]
        last_det = session.get('last_detection') or {}
        return {
            "exists": True,
            "hasStudent": session.get('student_sid') is not None,
            "studentId": session.get('student_id'),
            "facultyCount": len(session.get('faculty_members', [])),
            "lastUpdate": last_det.get('timestamp'),
            "alerts": last_det.get('proctoringAlerts', [])
        }
    else:
        return {"exists": False, "message": "Session not found"}
 
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    }
 
@app.get("/")
async def root():
    return {"message": "AI Proctoring Server is running - WebSocket only"}
 
# ======= Run server =======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5001, log_level="info")