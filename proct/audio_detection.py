import cv2
import numpy as np
import pyaudio
import winsound

frequency = 2500
duration = 1000

def audio_detection():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 2000  # Adjust this threshold based on your environment

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for audio...")

    # Flag to track whether suspicious sound has been detected
    suspicious_audio_detected = False

    while True:
        try:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if the audio exceeds the threshold (loud noise)
            if np.max(np.abs(audio_data)) > THRESHOLD and not suspicious_audio_detected:
                print("Suspicious audio detected!")

                # Beep Sound
                winsound.Beep(frequency, duration)

                #Set flag to True to indicate detection
                suspicious_audio_detected = True

                # Capture a frame from the camera
                # capture_and_save_frame()
            
            if np.max(np.abs(audio_data)) < THRESHOLD:
                suspicious_audio_detected = False

        except KeyboardInterrupt:
            break

    print("Stopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()

def capture_and_save_frame():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()

    if ret:
        # Save the frame or perform additional processing as needed
        cv2.imwrite("suspicious_frame.jpg", frame)
        print("Suspicious frame captured.")

    cap.release()

# if __name__ == "__main__":
#     audio_detection()

import sounddevice as sd
import numpy as np

# Settings
SAMPLE_RATE = 16000   # 16 kHz sample rate
DURATION = 0.5        # half a second per sample
THRESHOLD = 0.01      # adjust for sensitivity

def detect_audio():
    """Capture mic input and classify as Speaking / Silent"""

    try:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        # Compute RMS energy
        rms = np.sqrt(np.mean(np.square(audio)))

        if rms > THRESHOLD:
            return "Speaking"
        else:
            return "Silent"

    except Exception as e:
        return f"Audio Error: {e}"


if __name__ == "__main__":
    while True:
        status = detect_audio()
        print("[Audio]", status)
