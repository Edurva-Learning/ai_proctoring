import os
import sys
import time
import numpy as np

# ensure project root is importable when running from scripts/
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from recordings import save_frame_for_segment, _ensure_session_tmp

# Prepare
active_sessions = {}
session_id = 'testsession_smoke'
active_sessions[session_id] = {'student_id': 'user123'}

# Dummy frame
frame = np.full((480, 640, 3), 128, dtype=np.uint8)

# Save a few webcam frames
save_frame_for_segment(session_id, frame, active_sessions, 'webcam')
save_frame_for_segment(session_id, frame, active_sessions, 'webcam')

# Save a few screen frames
save_frame_for_segment(session_id, frame, active_sessions, 'screen')
save_frame_for_segment(session_id, frame, active_sessions, 'screen')

# Allow a short moment for any background tasks to schedule
time.sleep(0.5)

tmp = _ensure_session_tmp(session_id)
webcam_dir = os.path.join(tmp, 'webcam')
screen_dir = os.path.join(tmp, 'screen')

print('Session tmp:', tmp)
print('Webcam dir exists:', os.path.exists(webcam_dir))
print('Screen dir exists:', os.path.exists(screen_dir))

if os.path.exists(webcam_dir):
    print('Webcam files:', sorted(os.listdir(webcam_dir)))
if os.path.exists(screen_dir):
    print('Screen files:', sorted(os.listdir(screen_dir)))

# Clean up created files for smoke test
# (comment these lines if you want to inspect files afterwards)
# import shutil
# shutil.rmtree(tmp)
