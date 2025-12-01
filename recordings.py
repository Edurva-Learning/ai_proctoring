import os
import cv2
import shutil
import time
import subprocess
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import asyncio
from datetime import datetime

# Configuration defaults (can be overridden by env vars)
SEGMENT_SECONDS = int(os.environ.get('SEGMENT_SECONDS', '60'))
SEGMENT_FPS = int(os.environ.get('SEGMENT_FPS', '25'))
SEGMENT_MAX_WIDTH = int(os.environ.get('SEGMENT_MAX_WIDTH', '640'))
S3_BUCKET = os.environ.get('S3_BUCKET', 'edurva-courses')
S3_PREFIX = os.environ.get('S3_PREFIX', 'exam-recordings')

# Allow forcing the use of OpenCV encoder instead of ffmpeg via env var
FORCE_CV2 = str(os.environ.get('FORCE_CV2', '')).lower() in ('1', 'true', 'yes')

_s3_client = None
FFMPEG_CMD = None
# Tmp dir configuration and cleanup defaults
TMP_BASE = os.environ.get('SESSIONS_TMP_DIR') or os.path.join(os.path.dirname(__file__), 'tmp')
TMP_CLEANUP_INTERVAL = int(os.environ.get('TMP_CLEANUP_INTERVAL', str(60 * 60)))  # seconds between cleanup runs (default 1h)
TMP_CLEANUP_AGE = int(os.environ.get('TMP_CLEANUP_AGE', str(60 * 60)))  # session dirs older than this (seconds) will be removed


def _cleanup_session_tmp(session_id: str, base: str = None):
    """Remove the session tmp folder if it exists and is empty.

    This is a best-effort helper to remove leftover session folders once all
    frames/segments have been removed. It will only delete the specific
    session folder when it contains no files or subdirectories.
    """
    try:
        base_dir = base or TMP_BASE
        path = os.path.join(base_dir, session_id)
        if not os.path.exists(path):
            return False
        # Only remove if directory is empty
        if any(os.scandir(path)):
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception:
        return False


def cleanup_old_tmp(base: str = None, max_age_seconds: int = None, exclude_sessions: set | None = None):
    """Remove session tmp folders older than `max_age_seconds`.

    - `base`: base tmp directory; defaults to `TMP_BASE`.
    - `max_age_seconds`: age threshold; defaults to `TMP_CLEANUP_AGE`.
    - `exclude_sessions`: optional set of session ids to skip.
    """
    try:
        base_dir = base or TMP_BASE
        age = max_age_seconds or TMP_CLEANUP_AGE
        now = time.time()
        if not os.path.exists(base_dir):
            return []
        removed = []
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            sid = entry.name
            if exclude_sessions and sid in exclude_sessions:
                continue
            try:
                mtime = os.path.getmtime(entry.path)
                if (now - mtime) > age:
                    shutil.rmtree(entry.path, ignore_errors=True)
                    removed.append(entry.path)
            except Exception:
                continue
        return removed
    except Exception:
        return []


def start_periodic_tmp_cleanup(exclude_sessions_getter=None, base: str = None, interval: int = None, age: int = None):
    """Start a background daemon thread that periodically removes old tmp folders.

    - `exclude_sessions_getter`: optional callable returning an iterable of session ids to exclude.
    - `base`: tmp base directory to clean (defaults to `TMP_BASE`).
    - `interval`: seconds between runs (defaults to `TMP_CLEANUP_INTERVAL`).
    - `age`: age threshold in seconds (defaults to `TMP_CLEANUP_AGE`).
    """
    import threading

    def _loop():
        b = base or TMP_BASE
        itv = interval or TMP_CLEANUP_INTERVAL
        a = age or TMP_CLEANUP_AGE
        while True:
            try:
                exclude = None
                if callable(exclude_sessions_getter):
                    try:
                        ex = exclude_sessions_getter()
                        exclude = set(ex) if ex is not None else None
                    except Exception:
                        exclude = None
                cleanup_old_tmp(base=b, max_age_seconds=a, exclude_sessions=exclude)
            except Exception:
                pass
            time.sleep(itv)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _find_ffmpeg():
    global FFMPEG_CMD
    if FFMPEG_CMD:
        return FFMPEG_CMD
    env_path = os.environ.get('FFMPEG_PATH')
    if env_path and os.path.exists(env_path):
        FFMPEG_CMD = env_path
        return FFMPEG_CMD
    try:
        from shutil import which
        p = which('ffmpeg')
        if p:
            FFMPEG_CMD = p
            return FFMPEG_CMD
    except Exception:
        pass
    FFMPEG_CMD = None
    return None


def _encode_with_cv2(work_dir: str, out_file: str, fps: float):
    files = sorted([f for f in os.listdir(work_dir) if f.lower().endswith('.jpg')])
    if not files:
        raise RuntimeError('No frames to encode')
    first = cv2.imread(os.path.join(work_dir, files[0]))
    if first is None:
        raise RuntimeError('Failed to read first frame for cv2 encoding')
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError('cv2.VideoWriter failed to open (mp4 codec may be unavailable)')
    for fname in files:
        img = cv2.imread(os.path.join(work_dir, fname))
        if img is None:
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return True


def _ensure_s3_client():
    global _s3_client
    if _s3_client is None:
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
        _s3_client = boto3.client('s3', region_name=region)
    return _s3_client


def _ensure_session_tmp(session_id: str):
    base = TMP_BASE
    path = os.path.join(base, session_id)
    os.makedirs(path, exist_ok=True)
    return path


async def _segment_uploader_loop(session_id: str, user_id: str, active_sessions: dict, capture_type: str = 'webcam'):
    """Background loop that every SEGMENT_SECONDS checks for new frames for a specific capture_type,
    encodes them to a segment, and uploads to S3. Frames are stored in per-type subfolders under session tmp.
    """
    while session_id in active_sessions:
        await asyncio.sleep(SEGMENT_SECONDS)
        try:
            sess = active_sessions.get(session_id)
            if not sess:
                break

            tmp = _ensure_session_tmp(session_id)
            type_dir = os.path.join(tmp, capture_type)
            # indices and processed markers are maintained per-capture-type
            next_idx = sess.get('next_frame_index', {}).get(capture_type, 1)
            last_processed = sess.get('last_processed_index', {}).get(capture_type, 0)
            if next_idx - 1 <= last_processed:
                continue

            work = os.path.join(type_dir, 'work')
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work, exist_ok=True)

            copy_count = 0
            indices = list(range(last_processed + 1, next_idx))
            for i, src_idx in enumerate(indices, start=1):
                src = os.path.join(type_dir, f"frame_{src_idx:06d}.jpg")
                dst = os.path.join(work, f"frame_{i:06d}.jpg")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    copy_count += 1

            # extract timestamps for the slice (if available) to infer FPS
            all_times = sess.get('frame_times', {}).get(capture_type, [])
            frame_times_slice = [ts for (abs_idx, ts) in all_times if last_processed < abs_idx <= (next_idx - 1)]

            seg_idx = sess.get('segment_index', {}).get(capture_type, 0)
            print(f"🔍 Segment {capture_type} {seg_idx}: encoding indices {last_processed+1}-{next_idx-1}, copied={copy_count}, timestamps_entries={len(all_times)}, timestamps_used={len(frame_times_slice)}")

            if copy_count == 0:
                shutil.rmtree(work, ignore_errors=True)
                continue

            ts = int(time.time())
            out_file = os.path.join(type_dir, f"segment_{capture_type}_{ts}_{seg_idx}.mp4")

            inferred_fps = SEGMENT_FPS
            try:
                if frame_times_slice and len(frame_times_slice) >= 2:
                    duration = frame_times_slice[-1] - frame_times_slice[0]
                    frames_count = len(frame_times_slice)
                    if duration > 0.05:
                        inferred_fps = max(1.0, min(30.0, frames_count / duration))
                    else:
                        inferred_fps = SEGMENT_FPS
                    print(f"ℹ️ Segment {capture_type} {seg_idx}: frames={frames_count}, first_ts={frame_times_slice[0]:.3f}, last_ts={frame_times_slice[-1]:.3f}, duration={duration:.3f}s, inferred_fps={inferred_fps:.2f}")
                else:
                    inferred_fps = SEGMENT_FPS
                    print(f"ℹ️ Segment {capture_type} {seg_idx}: no timestamps available, using default fps={SEGMENT_FPS}")
            except Exception as ex:
                inferred_fps = SEGMENT_FPS
                print(f"⚠️ Segment {capture_type} {seg_idx}: error computing inferred_fps: {ex}, using fps={SEGMENT_FPS}")

            # If configured to force cv2, skip ffmpeg entirely
            USE_NVENC = str(os.environ.get('USE_NVENC', '')).lower() in ('1', 'true', 'yes')
            ffmpeg_path = None if FORCE_CV2 else _find_ffmpeg()

            if ffmpeg_path:
                # Choose encoder: prefer nvenc when requested, otherwise libx264
                encoder = 'h264_nvenc' if USE_NVENC else 'libx264'
                preset = os.environ.get('FFMPEG_PRESET', 'veryfast')
                crf = os.environ.get('FFMPEG_CRF', '23')

                cmd = [
                    ffmpeg_path, '-y', '-framerate', str(int(round(inferred_fps))), '-i', os.path.join(work, 'frame_%06d.jpg'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', encoder,
                    '-preset', preset,
                    '-crf', crf,
                    '-pix_fmt', 'yuv420p',
                    out_file
                ]

                try:
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"✅ ffmpeg encoded {capture_type} segment for session {session_id} (seg {seg_idx}) using {encoder}")
                except subprocess.CalledProcessError as cpe:
                    print(f"⚠️ ffmpeg encoding failed for session {session_id} ({capture_type}): returncode={cpe.returncode}")
                    if cpe.stdout:
                        print(f"ffmpeg stdout: {cpe.stdout}")
                    if cpe.stderr:
                        print(f"ffmpeg stderr: {cpe.stderr}")
                    # try cv2 fallback
                    try:
                        _encode_with_cv2(work, out_file, inferred_fps)
                        print(f"ℹ️ Used cv2 fallback after ffmpeg failure for session {session_id} ({capture_type})")
                    except Exception as ce:
                        print(f"⚠️ cv2 fallback encoding also failed for session {session_id} ({capture_type}): {ce}")
                        shutil.rmtree(work, ignore_errors=True)
                        continue
                except Exception as e:
                    print(f"⚠️ ffmpeg invocation error for session {session_id} ({capture_type}): {e}")
                    try:
                        _encode_with_cv2(work, out_file, inferred_fps)
                        print(f"ℹ️ Used cv2 fallback after ffmpeg invocation error for session {session_id} ({capture_type})")
                    except Exception as ce:
                        print(f"⚠️ cv2 fallback encoding also failed for session {session_id} ({capture_type}): {ce}")
                        shutil.rmtree(work, ignore_errors=True)
                        continue
            else:
                # No ffmpeg available or FORCE_CV2 requested: use OpenCV encoder
                try:
                    _encode_with_cv2(work, out_file, inferred_fps)
                    print(f"ℹ️ ffmpeg not used; cv2 fallback encoded {capture_type} segment for session {session_id}")
                except Exception as ce:
                    print(f"⚠️ Encoding failed for session {session_id} ({capture_type}): {ce}")
                    print("ℹ️ Install ffmpeg and ensure it's on PATH for more robust encoding. See https://ffmpeg.org/")
                    shutil.rmtree(work, ignore_errors=True)
                    continue

            try:
                await asyncio.to_thread(_upload_segment_to_s3, out_file, session_id, user_id, capture_type, seg_idx, sess)
            except Exception as e:
                print(f"⚠️ Upload failed for session {session_id} ({capture_type}): {e}")

            processed_up_to = next_idx - 1
            # persist per-type processed index
            sess.setdefault('last_processed_index', {})[capture_type] = processed_up_to
            try:
                if 'frame_times' in sess and sess['frame_times'].get(capture_type):
                    sess['frame_times'][capture_type] = [entry for entry in sess['frame_times'][capture_type] if entry[0] > processed_up_to]
            except Exception as e:
                print(f"⚠️ Failed trimming frame_times for session {session_id} ({capture_type}): {e}")
            sess.setdefault('segment_index', {})[capture_type] = seg_idx + 1
            active_sessions[session_id] = sess

            for src_idx in range(last_processed + 1, next_idx):
                p = os.path.join(type_dir, f"frame_{src_idx:06d}.jpg")
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            shutil.rmtree(work, ignore_errors=True)

            try:
                if os.path.exists(out_file):
                    os.remove(out_file)
            except Exception:
                pass

            # Attempt best-effort cleanup of the session tmp folder if it's empty
            try:
                # _cleanup_session_tmp will only delete when directory is empty
                _cleanup_session_tmp(session_id)
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Segment uploader error for session {session_id} ({capture_type}): {e}")

    if session_id in active_sessions:
        # clear uploader_running flag for this type
        try:
            active_sessions[session_id].setdefault('uploader_running', {})[capture_type] = False
        except Exception:
            active_sessions[session_id]['uploader_running'] = False


def _upload_segment_to_s3(filepath: str, session_id: str, user_id: str, capture_type: str, segment_idx: int, session_meta: dict = None):
    try:
        client = _ensure_s3_client()
        bucket = S3_BUCKET

        # prefer provided session_meta; fallback to empty dict
        sess = session_meta if isinstance(session_meta, dict) else {}
        # If caller wants exam/course ids, they should set them in active_sessions prior to upload
        # to keep this function generic.

        # Prefer explicit userId from call, otherwise look in session metadata (supports both camelCase and snake_case)
        user_part = (user_id or sess.get('userId') or sess.get('user_id') or sess.get('student_id') or 'unknown_user')
        course_part = sess.get('courseId') or sess.get('course_id') or 'unknown_course'
        exam_part = sess.get('examId') or sess.get('exam_id') or session_id
        date_part = datetime.utcnow().strftime('%Y%m%d')
        ts = int(time.time())
        # store under user_id/course_id/exam_id/<capture_type>/date/
        key = f"{S3_PREFIX}/{user_part}/{course_part}/{exam_part}/{session_id}/{capture_type}/{date_part}/{ts}_{segment_idx}.mp4"

        metadata = {
            'sessionId': session_id,
            # use the resolved user_part (prefers frontend userId/session metadata)
            'userId': user_part,
            'captureType': capture_type,
            'segmentIndex': str(segment_idx)
        }

        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                # debug: show key & metadata before upload
                print(f"⬆️ Uploading segment to s3://{bucket}/{key} metadata={metadata}")
                client.upload_file(
                    Filename=filepath,
                    Bucket=bucket,
                    Key=key,
                    ExtraArgs={
                        'ContentType': 'video/mp4',
                        'Metadata': metadata,
                        'ACL': 'private'
                    }
                )
                print(f"✅ Uploaded segment to s3://{bucket}/{key}")
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                return True
            except (BotoCoreError, ClientError) as e:
                print(f"⚠️ S3 upload attempt {attempts} failed for {filepath}: {e}")
                time.sleep(2 ** attempts)

        print(f"❌ Failed to upload {filepath} after {attempts} attempts")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected S3 upload error for {filepath}: {e}")
        return False


def save_frame_for_segment(session_id: str, frame, active_sessions: dict, capture_type: str = 'webcam'):
    """Downscale and save a frame as JPG into the session tmp folder with sequential index.
    Accepts `capture_type` ('webcam' or 'screen') so the uploader can tag and upload segments appropriately.
    Starts uploader background task if needed.
    """
    try:
        tmp = _ensure_session_tmp(session_id)
        sess = active_sessions.get(session_id, {})
        # ensure structure for per-type tracking
        sess.setdefault('next_frame_index', {})
        sess.setdefault('last_processed_index', {})
        sess.setdefault('segment_index', {})
        sess.setdefault('frame_times', {})
        sess.setdefault('uploader_running', {})

        # record capture type for the session (used by uploader)
        try:
            sess['capture_type'] = capture_type
        except Exception:
            pass

        type_dir = os.path.join(tmp, capture_type)
        os.makedirs(type_dir, exist_ok=True)

        idx = sess['next_frame_index'].get(capture_type, 1)

        h, w = frame.shape[:2]
        if w > SEGMENT_MAX_WIDTH:
            new_w = SEGMENT_MAX_WIDTH
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

        fname = os.path.join(type_dir, f"frame_{idx:06d}.jpg")
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with open(fname, 'wb') as f:
            f.write(buf.tobytes())

        sess['next_frame_index'][capture_type] = idx + 1
        times = sess['frame_times'].get(capture_type, [])
        times.append((idx, time.time()))
        sess['frame_times'][capture_type] = times
        active_sessions[session_id] = sess

        # start uploader for this capture_type if not running
        if not sess['uploader_running'].get(capture_type):
            sess['uploader_running'][capture_type] = True
            sess['last_processed_index'][capture_type] = sess['last_processed_index'].get(capture_type, 0)
            sess['segment_index'][capture_type] = sess['segment_index'].get(capture_type, 0)
            active_sessions[session_id] = sess
            # Start uploader: use existing running loop if present, otherwise start a background thread
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_segment_uploader_loop(session_id, sess.get('student_id'), active_sessions, capture_type))
            except RuntimeError:
                # no running loop in this process (e.g., smoke test); run uploader in a background thread event loop
                import threading

                def _bg():
                    try:
                        asyncio.run(_segment_uploader_loop(session_id, sess.get('student_id'), active_sessions, capture_type))
                    except Exception as _e:
                        print(f"⚠️ Background uploader error for session {session_id} ({capture_type}): {_e}")

                t = threading.Thread(target=_bg, daemon=True)
                t.start()
    except Exception as e:
        print(f"⚠️ Failed saving frame for session {session_id} ({capture_type}): {e}")


def combine_s3_segments_to_local(session_meta: dict, capture_type: str, work_dir_root: str = None):
    """Download and concatenate S3 segments for the given session_meta and capture_type.

    session_meta should contain keys: userId/user_id/student_id, courseId/course_id, examId/exam_id
    Returns local path to combined mp4. Raises RuntimeError on failure.
    """
    client = _ensure_s3_client()

    user_part = (session_meta.get('userId') or session_meta.get('user_id') or session_meta.get('student_id') or 'unknown_user')
    course_part = session_meta.get('courseId') or session_meta.get('course_id') or 'unknown_course'
    exam_part = session_meta.get('examId') or session_meta.get('exam_id') or session_meta.get('session_id') or 'unknown_exam'

    prefix = f"{S3_PREFIX}/{user_part}/{course_part}/{exam_part}/{capture_type}/"

    objs = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for item in page.get('Contents', []) or []:
            key = item.get('Key')
            if key and key.lower().endswith('.mp4'):
                objs.append((key, item.get('LastModified')))

    if not objs:
        raise RuntimeError(f'No segments found on S3 for prefix {prefix}')

    # sort by key (lexicographic; timestamps in filenames will order)
    objs.sort(key=lambda x: x[0])

    if not work_dir_root:
        work_dir_root = TMP_BASE
    work_dir = os.path.join(work_dir_root, f"concat_{user_part}_{exam_part}_{capture_type}")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    local_files = []
    for i, (key, _) in enumerate(objs, start=1):
        local_path = os.path.join(work_dir, f"seg_{i:04d}.mp4")
        try:
            client.download_file(S3_BUCKET, key, local_path)
            local_files.append(local_path)
        except Exception as e:
            print(f"⚠️ Failed downloading {key}: {e}")

    if not local_files:
        raise RuntimeError('Failed to download any segments')

    ff = _find_ffmpeg()
    out_file = os.path.join(work_dir, f"combined_{capture_type}_{int(time.time())}.mp4")

    if not ff:
        raise RuntimeError('ffmpeg not available on server; install ffmpeg to enable concatenation')

    # create concat list file
    list_txt = os.path.join(work_dir, 'files.txt')
    with open(list_txt, 'w', encoding='utf-8') as lf:
        for p in local_files:
            lf.write("file '{}'\n".format(p))

    cmd = [ff, '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy', out_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ ffmpeg concatenated {len(local_files)} segments into {out_file}")
        return out_file
    except subprocess.CalledProcessError as cpe:
        # try a re-encode approach
        print(f"⚠️ ffmpeg concat failed: {cpe.stderr}\nTrying re-encode fallback")
        cmd2 = [ff, '-y']
        for p in local_files:
            cmd2 += ['-i', p]
        cmd2 += ['-filter_complex', f"concat=n={len(local_files)}:v=1:a=1[outv][outa]", '-map', '[outv]', '-map', '[outa]', out_file]
        try:
            subprocess.run(cmd2, check=True, capture_output=True, text=True)
            print(f"✅ ffmpeg re-encoded concat created {out_file}")
            return out_file
        except Exception as e:
            raise RuntimeError(f'ffmpeg re-encode concat failed: {e}')
