import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
from math import dist

# ================== AUDIO SETUP ==================
pygame.mixer.init()

# Alarm sound (for drowsiness)
pygame.mixer.music.load("alarm.wav")
pygame.mixer.music.set_volume(1.0)

# Face not visible warning sound
face_warning_sound = pygame.mixer.Sound("face_warning.wav")
face_warning_sound.set_volume(1.0)

# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================== VARIABLES ==================
EAR_THRESHOLD = 0.21
DROWSY_TIME = 2.5          # seconds eyes closed
ALARM_DURATION = 8         # seconds alarm plays

eye_closed_start = None
alarm_on = False
alarm_start = None

# Face not visible warning control
face_missing_start = None
last_face_warning_time = 0
FACE_MISSING_TIME = 2.0        # wait before warning
FACE_WARNING_INTERVAL = 5.0    # repeat warning every 5 sec

# ================== FUNCTIONS ==================
def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
print("Press Q to quit")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    result = face_mesh.process(rgb)

    current_time = time.time()

    # ================= FACE NOT VISIBLE WARNING =================
    if not result.multi_face_landmarks:
        if face_missing_start is None:
            face_missing_start = current_time
            last_face_warning_time = 0

        elif current_time - face_missing_start >= FACE_MISSING_TIME:
            if current_time - last_face_warning_time >= FACE_WARNING_INTERVAL:
                face_warning_sound.play()
                last_face_warning_time = current_time

        # Pause drowsiness detection when face missing
        eye_closed_start = None

    else:
        # Face detected again → reset warning
        face_missing_start = None
        last_face_warning_time = 0
        face_warning_sound.stop()

        # ================= DROWSINESS DETECTION =================
        for face in result.multi_face_landmarks:
            landmarks = face.landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            # Eyes closed
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time

                elif current_time - eye_closed_start >= DROWSY_TIME:
                    if not alarm_on:
                        pygame.mixer.music.play(-1)
                        alarm_on = True
                        alarm_start = current_time

            # Eyes open → reset
            else:
                eye_closed_start = None
                if alarm_on:
                    pygame.mixer.music.stop()
                    alarm_on = False
                    alarm_start = None

            # Stop alarm after fixed duration
            if alarm_on and alarm_start is not None:
                if current_time - alarm_start >= ALARM_DURATION:
                    pygame.mixer.music.stop()
                    alarm_on = False
                    alarm_start = None
                    eye_closed_start = None

            # Show EAR on screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ================= DISPLAY =================
    if alarm_on:
        cv2.putText(frame, "DROWSY! WAKE UP!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("High Accuracy Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
